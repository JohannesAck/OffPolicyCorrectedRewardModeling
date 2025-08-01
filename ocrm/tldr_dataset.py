import copy
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import tyro
from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from rich.pretty import pprint
from transformers import AutoTokenizer

api = HfApi()


"""
poetry run python -i summarize_from_feedback_details/tldr_dataset.py \
    --base_model=Qwen/Qwen2.5-1.5B \
    --tldr_params.max_sft_response_length=53 \
    --tldr_params.max_sft_query_response_length=562 \
    --tldr_params.max_rm_response_length=169 \
    --tldr_params.max_rm_query_response_length=638 \
    --cnndm_params.max_rm_response_length=155 \
    --cnndm_params.max_rm_query_response_length=2021 \
    --push_to_hub \

poetry run python -i summarize_from_feedback_details/tldr_dataset.py \
    --base_model=Qwen/Qwen2.5-1.5B \
    --tldr_params.max_sft_response_length=53 \
    --tldr_params.max_sft_query_response_length=562 \
    --tldr_params.max_rm_response_length=169 \
    --tldr_params.max_rm_query_response_length=638 \
    --cnndm_params.max_rm_response_length=155 \
    --cnndm_params.max_rm_query_response_length=2021 \
    --push_to_hub \
    --tldr_params.padding="empty_space" \
    --cnndm_params.padding="empty_space" \
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
            format_str="SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:",
            truncate_field="post",
            truncate_text="\n",
            padding="pad_token",
            pad_side="left",
            max_sft_response_length=53,
            max_sft_query_response_length=562,
            max_rm_response_length=169,
            max_rm_query_response_length=638,
        )
    )
    cnndm_params: TaskQueryHParams = field(
        default_factory=lambda: TaskQueryHParams(
            length=2047 - 128,
            format_str="Article:\n{article}\n\nTL;DR:\n",
            truncate_field="article",
            truncate_text="\n",
            padding="pad_token",
            pad_side="left",
            max_rm_response_length=155,
            max_rm_query_response_length=2021,
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


def process_query(query_info: Dict[str, str], *, encoder, hparams: TaskQueryHParams, pad_sequence=None):
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
    sft_ds = load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered")

    def process_query_data(x):
        # the `x['summary']` in `vwxyzjn/summarize_from_feedback_tldr_3_filtered`
        # DOES NOT HAVE a leading space so we are adding the leading space and
        # `<|endoftext|>` token
        reference_response = f" {x['summary']}<|endoftext|>"
        y = {
            **process_query(x, encoder=tokenizer, hparams=args.tldr_params),
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

    sft_ds = sft_ds.map(process_query_data, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count())
    if args.push_to_hub:
        sft_dataset_hf_path = f"{args.hf_entity}/summarize_from_feedback_tldr_3_filtered_oai_preprocessingQWEN_{timestamp}"
        sft_ds.push_to_hub(sft_dataset_hf_path)
        sft_card = RepoCard.load(sft_dataset_hf_path, repo_type="dataset")
        sft_card.text = f"""\
# TL;DR SFT Dataset for OpenAI's [Summarize from Feedback](https://openai.com/blog/summarization/) task

The dataset is directly taken from https://github.com/openai/summarize-from-feedback/tree/700967448d10004279f138666442bf1497d0e705#reddit-tldr-dataset

These columns are taken directly from the aforementioned dataset:

* **id**: unique identifier for the post
* **subreddit**: subreddit the post was taken from
* **title**: title of the post
* **post**: body of the post
* **summary**: summary of the post
* **reference_response**: reference response for the post

These columns are added by this preprocessing script:
* **query**: length-limited query for summarization: OAI pre-processes the main text (title + subreddit + post), ensuring it has only 512 tokens; if the main text is too long, then it tries to truncate at the last `\n`. If it's too short it pads the main text ([summarize_from_feedback/tasks.py#L98-L165](https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/summarize_from_feedback/tasks.py#L98-L165)). Padding is either space or `[PAD]` token (see Args below).
* **query_token**: tokenized version of `query`
* **reference_response_token**: tokenized version of `reference_response`
* **reference_response_token_len**: length of `reference_response_token`
* **query_reference_response**: concatenation of `query.strip()` and `reference_response`
* **query_reference_response_token**: tokenized version of `query_reference_response`, up to `max_sft_query_response_length` tokens
* **query_reference_response_token_len**: length of `query_reference_response_token`


# Args

```python
{pformat(vars(args))}
```
"""
        sft_card.push_to_hub(sft_dataset_hf_path, repo_type="dataset")


    lens_before = {k: len(sft_ds[k]) for k in sft_ds.keys()}
    print(f'SFT Dataset len before filtering {lens_before}')
    sft_ds = sft_ds.filter(lambda x: x["query_reference_response_token_len"] <= args.tldr_params.max_sft_query_response_length and x["reference_response_token_len"] <= args.tldr_params.max_sft_response_length)
    lens_after = {k: len(sft_ds[k]) for k in sft_ds.keys()}
    print(f'SFT Dataset len after filtering {lens_after}')  # this removes 70 / 116000 samples, should be fine

    ####################################
    # visualize token length distribution
    ####################################
    calculated_tldr_params = TaskQueryHParams(
        max_sft_query_response_length=0,
        max_sft_response_length=0,
    )

    os.makedirs("dataset_visuals", exist_ok=True)
    num_sft_visuals = 2
    num_label_visuals = 5
    num_subplots = len(sft_ds) * num_sft_visuals
    num_cols = 3
    print(f"{num_subplots=}")
    fig, axs = plt.subplots(ceil_div(num_subplots, num_cols), num_cols, figsize=(16, 16))
    axs = axs.flatten()
    j = 0
    for _, key in enumerate(sft_ds.keys()):
        df = sft_ds[key].to_pandas()
        axs[j].hist(df["reference_response_token_len"], bins=100)
        axs[j].set_title(f"{key} split: reference response token length\nmax_length={max(df['reference_response_token_len'])}")
        axs[j + 1].hist(df["query_reference_response_token_len"], bins=100)
        axs[j + 1].set_title(
            f"{key} split: query.strip() + reference response token length\nmax_length={max(df['query_reference_response_token_len'])}"
        )
        calculated_tldr_params.max_sft_response_length = max(
            calculated_tldr_params.max_sft_response_length, max(df["reference_response_token_len"])
        )
        calculated_tldr_params.max_sft_query_response_length = max(
            calculated_tldr_params.max_sft_query_response_length, max(df["query_reference_response_token_len"])
        )
        j += num_sft_visuals
    offset = len(sft_ds)
    fig.suptitle(f"{args.base_model} Tokenizer: Token length distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/token_len.png")

    pprint({"calculated_tldr_params": calculated_tldr_params})
    if args.check_length_correctness:
        assert calculated_tldr_params.max_sft_response_length <= args.tldr_params.max_sft_response_length
        assert calculated_tldr_params.max_sft_query_response_length <= args.tldr_params.max_sft_query_response_length
        print("✨ calculated lenghts are ok!")


    if args.push_to_hub:
        # upload the `dataset_visuals`
        api.upload_folder(
            folder_path="dataset_visuals",
            path_in_repo="dataset_visuals",
            repo_id=f"{args.hf_entity}/summarize_from_feedback_oai_preprocessingQWEN_{timestamp}",
            repo_type="dataset",
        )
        # upload current file
        print(f"{__file__=}")
        api.upload_file(
            path_or_fileobj=__file__,
            path_in_repo="create_dataset.py",
            repo_id=f"{args.hf_entity}/summarize_from_feedback_oai_preprocessingQWEN_{timestamp}",
            repo_type="dataset",
        )
        print(f"✨ Pushed to hub: https://huggingface.co/datasets/{sft_dataset_hf_path}")

