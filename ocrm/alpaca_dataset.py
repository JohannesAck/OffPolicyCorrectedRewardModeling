import copy
from functools import partial
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import tyro
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from rich.pretty import pprint
from transformers import AutoTokenizer

api = HfApi()


"""
poetry run python -i summarize_from_feedback_details/tldr_dataset.py \
    --base_model=EleutherAI/pythia-1b-deduped \
    --tldr_params.max_sft_response_length=106 \
    --tldr_params.max_sft_query_response_length=615 \

"""


@dataclass
class TaskQueryHParams:
    length: Optional[int] = None
    format_str: Optional[str] = None
    truncate_field: Optional[str] = None
    truncate_text: Optional[str] = None
    padding: Optional[Literal["empty_space", "pad_token"]] = None
    pad_token: Optional[str] = None
    pad_side: Optional[str] = None
    max_sft_response_length: Optional[int] = None
    max_sft_query_response_length: Optional[int] = None
    max_rm_response_length: Optional[int] = None
    max_rm_query_response_length: Optional[int] = None


@dataclass
class Args:
    base_model: str = "EleutherAI/pythia-1b-deduped"  #  "gpt2"
    hf_entity: str = None
    push_to_hub: bool = False
    check_length_correctness: bool = True
    debug: bool = False
    tldr_params: TaskQueryHParams = field(
        default_factory=lambda: TaskQueryHParams(
            length=512,
            format_str="INSTRUCTION: {instruction} \n\nINPUT: {input}\n\nREPLY:",
            truncate_field="input",
            truncate_text="\n",
            padding="pad_token",
            pad_side="left",
            max_sft_response_length=106,
            max_sft_query_response_length=615,
        )
    )


def _ensure_length(toks, l, pad_sequence=None, pad_side=None, truncate_side=None):
    assert pad_side in (None, "left", "right")
    assert truncate_side in (None, "left", "right")
    if len(toks) < l:
        assert pad_sequence is not None
        pad_amt = l - len(toks)
        assert len(pad_sequence) >= pad_amt, f"{len(pad_sequence)} < {pad_amt}"
        if pad_side is None:
            assert len(toks) == l, f"Needed to pad! {len(toks)} < {l}"
            return toks
        elif pad_side == "left":
            return pad_sequence[-pad_amt:] + toks
        else:
            assert pad_side == "right"
            return toks + pad_sequence[:pad_amt]
    if truncate_side is None:
        assert len(toks) == l, f"Needed to truncate! {len(toks)} > {l}"
        return toks
    elif truncate_side == "left":
        return toks[-l:]
    else:
        assert truncate_side == "right"
        return toks[:l]


def _get_query_padding_for_task(encoder, hparams: TaskQueryHParams):
    return hparams.pad_token * hparams.length


def process_query(query_info: Dict[str, str], *, encoder, hparams: TaskQueryHParams, pad_sequence=None, istestdata=False):
    if pad_sequence is None:
        pad_sequence = _get_query_padding_for_task(encoder, hparams)
    if isinstance(query_info, str):
        query_info = dict(query=query_info)
    else:
        # copy to avoid mutating input
        query_info = dict(**query_info)

    format_str = hparams.format_str or "{query}"
    query_tokens = encoder.encode(format_str.format(**query_info))
    truncate_field = hparams.truncate_field or "query"

    if truncate_field not in query_info:
        raise ValueError(f"Could not truncate field {truncate_field}, found fields: {query_info.keys()}!")
    while len(query_tokens) > hparams.length:
        if not len(query_info[truncate_field]):
            if istestdata:
                print(f"query too long query_info: {query_info}")
            else:
                raise ValueError("Could not truncate enough!")

        i = -1  # default to just remove one character
        if hparams.truncate_text:
            try:
                i = query_info[truncate_field].rindex(hparams.truncate_text)
            except ValueError:
                pass
        query_info[truncate_field] = query_info[truncate_field][:i]
        query_tokens = encoder.encode(format_str.format(**query_info))

    query_token = _ensure_length(query_tokens, hparams.length, pad_side=hparams.pad_side, pad_sequence=pad_sequence)
    query = encoder.decode(query_token, skip_special_tokens=True).lstrip()
    return dict(
        query_token=query_token,
        query=query,
    )


def ceil_div(a, b):
    return (a - 1) // b + 1


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.hf_entity is None:
        args.hf_entity = api.whoami()["name"]
        assert isinstance(args.hf_entity, str)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # post init
    if args.tldr_params.padding == "empty_space":
        args.tldr_params.pad_token = tokenizer.encode(" ")
    else:
        args.tldr_params.pad_token = [tokenizer.pad_token_id]
    pprint(args)
    timestamp = int(time.time())

    sft_split = load_dataset('json', data_files='alpaca_instructions/sft.json')['train']
    unlabeled_split = load_dataset('json', data_files='alpaca_instructions/unlabeled.json')['train']
    preference_split = load_dataset('json', data_files='alpaca_instructions/preference.json')['train']
    train_ds = concatenate_datasets([sft_split, unlabeled_split, preference_split])

    validation_ds = load_dataset('json', data_files='alpaca_instructions/val.json')['train']
    test_ds = load_dataset('json', data_files='alpaca_instructions/alpaca_farm_evaluation.json')['train']

    def process_query_data(x, istestdata=False):
        # the `x['summary']` in `vwxyzjn/summarize_from_feedback_tldr_3_filtered`
        # DOES NOT HAVE a leading space so we are adding the leading space and
        # `<|endoftext|>` token
        reference_response = f" {x['output']}<|endoftext|>"
        y = {
            **process_query(x, encoder=tokenizer, hparams=args.tldr_params, istestdata=istestdata),
            "reference_response": reference_response,
            "reference_response_token": tokenizer.encode(
                reference_response,
                padding="max_length",
                max_length=args.tldr_params.max_sft_response_length,
                truncation=True,
            ),
            "reference_response_token_len": len(tokenizer.encode(reference_response)),
        }
        y["query_reference_response"] = y["query"].strip() + y["reference_response"]
        # if padding is space, then we can just concatenate the tokens
        if args.tldr_params.padding == "empty_space":
            y["query_reference_response_token"] = y["query_token"] + y["reference_response_token"]
        else:
            y["query_reference_response_token"] = tokenizer.encode(
                y["query_reference_response"],
                padding="max_length",
                max_length=args.tldr_params.max_sft_query_response_length,
                truncation=True,
            )
        y["query_reference_response_token_response_label"] = copy.deepcopy(y["query_reference_response_token"])
        unpadded_query_token = [token for token in y["query_token"] if token != tokenizer.pad_token_id]
        y["query_reference_response_token_response_label"][:len(unpadded_query_token)] = [tokenizer.pad_token_id for _ in range(len(unpadded_query_token))]
        y["query_reference_response_token_len"] = len(tokenizer.encode(y["query_reference_response"]))
        return y

    train_ds = train_ds.map(process_query_data, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count())
    print(f'Train Dataset len before filtering {len(train_ds)}')
    train_ds = train_ds.filter(lambda x: x["query_reference_response_token_len"] <= args.tldr_params.max_sft_query_response_length and x["reference_response_token_len"] <= args.tldr_params.max_sft_response_length)
    print(f'Train Dataset len after filtering {len(train_ds)}')

    validation_ds = validation_ds.map(process_query_data, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count())
    print(f'Validation Dataset len before filtering {len(validation_ds)}')
    validation_ds = validation_ds.filter(lambda x: x["query_reference_response_token_len"] <= args.tldr_params.max_sft_query_response_length and x["reference_response_token_len"] <= args.tldr_params.max_sft_response_length)
    print(f'Validation Dataset len after filtering {len(validation_ds)}')

    # process_query_test = partial(process_query_data, istestdata=True)
    print(f'Test Dataset len before {len(test_ds)}')
    test_ds_filtered = test_ds.filter(lambda x: (len(x['instruction']) + len(x['input']) + len(x['output'])) <= 2000)
    print(f'Test Dataset len after character filter {len(test_ds_filtered)}')
    test_ds_filtered = test_ds_filtered.map(process_query_data, load_from_cache_file=False, num_proc=4)
    test_ds_filtered = test_ds_filtered.filter(lambda x: x["query_reference_response_token_len"] <= args.tldr_params.max_sft_query_response_length and x["reference_response_token_len"] <= args.tldr_params.max_sft_response_length)
    print(f'Test Dataset len after token filter {len(test_ds_filtered)}')

    test_ds_filtered = test_ds_filtered.remove_columns(['datasplit', 'dataset','generator','sample_mode'])
    merged_ds = DatasetDict({'train': train_ds, 'validation': validation_ds, 'test': test_ds_filtered})

    if args.push_to_hub:
        merged_dataset_hf_path = f"{args.hf_entity}/alpaca_gemma2_sft{timestamp}"
        merged_ds.push_to_hub(merged_dataset_hf_path)
        merged_card = RepoCard.load(merged_dataset_hf_path, repo_type="dataset")
        merged_card.text = f"""\
# Alpaca-instructions dataset in a format suitable for TLDR code by Costa Huang

Filtered to only include examples where the sum of the token length of the query and reference response is less than or equal to {args.tldr_params.max_sft_query_response_length} 
and the token length of the reference response is less than or equal to {args.tldr_params.max_sft_response_length}.

Validation dataset is also filtered to the max lenghts.
Test split is alpaca_farm_evaluation and is also filtered. (some of the examples are reaaaaally long)

see Costa's code at
https://github.com/vwxyzjn/summarize_from_feedback_details

```python
{pformat(vars(args))}
```
"""
        merged_card.push_to_hub(merged_dataset_hf_path, repo_type="dataset")


    ####################################
    # visualize token length distribution
    ####################################
    calculated_tldr_params = TaskQueryHParams(
        max_sft_query_response_length=0,
        max_sft_response_length=0,
        max_rm_response_length=0,
        max_rm_query_response_length=0,
    )
    calculated_cnndm_params = TaskQueryHParams(
        max_rm_query_response_length=0,
        max_rm_response_length=0,
    )

    os.makedirs("dataset_visuals", exist_ok=True)
    num_sft_visuals = 2
    num_label_visuals = 5
    num_subplots = 1
    num_cols = 3
    print(f"{num_subplots=}")
    fig, axs = plt.subplots(ceil_div(num_subplots, num_cols), num_cols, figsize=(16, 6))
    axs = axs.flatten()
    j = 0
    df = train_ds.to_pandas()
    axs[j].hist(df["reference_response_token_len"], bins=100)
    axs[j].set_title(f"split: reference response token length\nmax_length={max(df['reference_response_token_len'])}")
    axs[j + 1].hist(df["query_reference_response_token_len"], bins=100)
    axs[j + 1].set_title(
        f"split: query.strip() + reference response token length\nmax_length={max(df['query_reference_response_token_len'])}"
    )
    calculated_tldr_params.max_sft_response_length = max(
        calculated_tldr_params.max_sft_response_length, max(df["reference_response_token_len"])
    )
    calculated_tldr_params.max_sft_query_response_length = max(
        calculated_tldr_params.max_sft_query_response_length, max(df["query_reference_response_token_len"])
    )
    j += num_sft_visuals
    offset = len(train_ds)
    fig.suptitle(f"{args.base_model} Tokenizer: Token length distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/token_len.png")

    pprint({"calculated_tldr_params": calculated_tldr_params})
    if args.check_length_correctness:
        assert calculated_tldr_params.max_sft_response_length == args.tldr_params.max_sft_response_length
        assert calculated_tldr_params.max_sft_query_response_length == args.tldr_params.max_sft_query_response_length
        print("✨ calculated lenghts are ok!")



    if args.push_to_hub:
        # upload the `dataset_visuals`
        api.upload_folder(
            folder_path="dataset_visuals",
            path_in_repo="dataset_visuals",
            repo_id=merged_dataset_hf_path,
            repo_type="dataset",
        )
        # upload current file
        print(f"{__file__=}")
        api.upload_file(
            path_or_fileobj=__file__,
            path_in_repo="create_dataset.py",
            repo_id=merged_dataset_hf_path,
            repo_type="dataset",
        )
        print(f"✨ Pushed to hub: https://huggingface.co/datasets/{merged_dataset_hf_path}")
