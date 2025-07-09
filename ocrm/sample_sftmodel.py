import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import broadcast, gather_object, tqdm
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

torch.set_printoptions(precision=4, sci_mode=False)
api = HfApi()
INVALID_LOGPROB = 1.0


@dataclass
class PpoHParams:
    nminibatches: int = 1
    noptepochs: int = 4
    vf_coef: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = False
    kl_coef: float = 0.05


@dataclass
class Args:
    # common args
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """Whether to use cuda if available."""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    local_batch_size: Optional[int] = 16
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""

    # other args
    base_model: str = "EleutherAI/pythia-1b"
    """the name of the pretrained model to use"""
    sft_model_path: str = "models/EleutherAI/pythia-1b-deduped/sft_model_150"
    """the name of the pretrained model to use"""
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    """the query dataset"""
    response_length: int = 53
    """the length of the response"""
    truncate_token: Literal["eos"] = "eos"
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    sampling_epochs: int = 1
    """the number of epochs to sample for"""
    num_samples: Optional[int] = None
    """total number of samples to generate. Epoch is ignored if this is set"""
    out_name: str = "dataset_full"
    """the name of the output file, saved in the sft_model directory"""


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q

def parse_args() -> Args:
    args = tyro.cli(Args)
    return args


# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, critic) -> None:
        super().__init__()
        self.policy = policy
        self.critic = critic

    def forward(self, **kwargs):
        return self.policy(**kwargs), self.critic(**kwargs)



def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        cache_implementation='static',
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer, responses):
    trunc_idxs = first_true_indices(responses == args.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def forward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator(even_batches=False)
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    # load dataset
    dataset = load_dataset(args.query_dataset, split="train")
    dataset = dataset.with_format("torch", columns=["query_token", "reference_response_token", "reference_response"])
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size * torch.cuda.device_count(), shuffle=False)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if args.truncate_token == "eos":
        args.truncate_token_id = tokenizer.eos_token_id

    console = Console(force_terminal=True)

    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True
    model_config = AutoConfig.from_pretrained(args.base_model)

    ref_policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, config=model_config, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, config=model_config, trust_remote_code=True)

    for module in [policy, ref_policy]:
        disable_dropout(module)
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    model = PolicyAndValueWrapper(policy, policy)

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, dataloader = accelerator.prepare(model, dataloader)
    torch.manual_seed(local_seed)  # reset the local seed again

    def repeat_generator():
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
    ref_policy = ref_policy.to(device)

    generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        min_new_tokens=args.response_length,
        temperature=(args.temperature + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    accelerator.print("===training policy===")
    global_step = 0
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        total_query_ids = []
        total_decode_queries = []
        total_decode_responses1 = []
        total_response_ids1 = []
        total_query_response1_ids = []
        total_logprob1 = []
        total_sumlogprob1 = []
        total_decode_responses2 = []
        total_response_ids2 = []
        total_query_response2_ids = []
        total_logprob2 = []
        total_sumlogprob2 = []
        total_reference_response = []

        debug = False
        sampling_epochs = args.sampling_epochs
        if args.num_samples is not None:
            sampling_epochs = 1
        for epoch in range(sampling_epochs):
            for i, data in tqdm(enumerate(dataloader), desc=f"Generating Epoch {epoch}/{args.sampling_epochs}",
                                 total=len(dataloader), dynamic_ncols=True):
                query = data["query_token"].to(device)
                context_length = query.shape[1]

                query_response1, logits1 = generate(
                    accelerator.unwrap_model(model).policy,
                    query,
                    tokenizer,
                    generation_config,
                )
                response1 = query_response1[:, context_length:]

                # use the logits during generation directly, instead of using the following
                all_logprob1 = F.log_softmax(logits1, dim=-1)
                logprob1 = torch.gather(all_logprob1, 2, response1.unsqueeze(-1)).squeeze(-1)
                if not debug:
                    del logits1, all_logprob1

                if debug:
                    ref_output1 = forward(ref_policy, query_response1, tokenizer)
                    ref_logits1 = ref_output1.logits[:, context_length - 1 : -1]
                    ref_logits1 /= args.temperature + 1e-7
                    ref_all_logprob1 = F.log_softmax(ref_logits1, dim=-1)
                    ref_logprob1 = torch.gather(ref_all_logprob1, 2, response1.unsqueeze(-1)).squeeze(-1)
                    # del ref_output, ref_logits, ref_all_logprob
                    assert torch.allclose(logprob1, ref_logprob1, rtol=1e-1), "logprob mismatch"

                # Response Processing 1. truncate response after the first occurrence of `truncate_token_id`
                postprocessed_response1 = truncate_response(args, tokenizer, response1)
                decode_responses1 = tokenizer.batch_decode(
                    postprocessed_response1,
                    skip_special_tokens=True,
                )
                eos_positions1 = torch.nonzero(response1 == tokenizer.eos_token_id)
                logprob1_copy = logprob1.clone()
                for b, x in eos_positions1:
                    logprob1_copy[b,x + 1:] = 0.0  # leave the eos token's logprob intact, just zero afterward
                sum_logprob1 = logprob1_copy.sum(dim=1)


                query_response2, logits2 = generate(
                    accelerator.unwrap_model(model).policy,
                    query,
                    tokenizer,
                    generation_config,
                )
                response2 = query_response2[:, context_length:]

                # use the logits during generation directly, instead of using the following
                all_logprob2 = F.log_softmax(logits2, dim=-1)
                logprob2 = torch.gather(all_logprob2, 2, response2.unsqueeze(-1)).squeeze(-1)
                del logits2, all_logprob2
                
                postprocessed_response2 = truncate_response(args, tokenizer, response2)
                decode_responses2 = tokenizer.batch_decode(
                    postprocessed_response2,
                    skip_special_tokens=True,
                )
                eos_positions2 = torch.nonzero(response2 == tokenizer.eos_token_id)
                logprob2_copy = logprob2.clone()
                for b, x in eos_positions2:
                    logprob2_copy[b,x + 1:] = 0.0  # leave the eos token's logprob intact, just zero afterward
                sum_logprob2 = logprob2_copy.sum(dim=1)


                decode_queries = tokenizer.batch_decode(query)
                
                total_query_ids.extend(query.cpu().numpy())
                total_decode_queries.extend(decode_queries)
                total_response_ids1.extend(response1.cpu().numpy())
                total_query_response1_ids.extend(query_response1.cpu().numpy())
                total_decode_responses1.extend(decode_responses1)
                total_logprob1.extend(logprob1.cpu().numpy())
                total_sumlogprob1.extend(sum_logprob1.cpu().numpy())
                total_response_ids2.extend(response2.cpu().numpy())
                total_query_response2_ids.extend(query_response2.cpu().numpy())
                total_decode_responses2.extend(decode_responses2)
                total_logprob2.extend(logprob2.cpu().numpy())
                total_sumlogprob2.extend(sum_logprob2.cpu().numpy())
                total_reference_response.extend(data["reference_response"])

                if args.num_samples is not None and len(total_decode_queries) * accelerator.num_processes >= args.num_samples:
                    break

            
    data_dict = {
                "query": total_decode_queries,
                "query_token": total_query_ids,
                "response1": total_decode_responses1,
                "response1_token": total_response_ids1,
                "query_response1_token": total_query_response1_ids,
                "logprob1": total_logprob1,
                "sumlogprob1": total_sumlogprob1,
                "response2": total_decode_responses2,
                "response2_token": total_response_ids2,
                "query_response2_token": total_query_response2_ids,
                "logprob2": total_logprob2,
                "sumlogprob2": total_sumlogprob2,
                "reference_response": total_reference_response,
            }
    accelerator.wait_for_everyone()
    data_dict_gathered = {k: gather_object(v) for k, v in data_dict.items()}
    data_dict_gathered['gen_id'] = np.arange(len(data_dict_gathered['query']))

    if accelerator.is_main_process:
        save_pickle = False
        if save_pickle:
            out_fp = os.path.join(args.sft_model_path, args.out_name + f".npy")
            np.save(out_fp, data_dict_gathered)
            print(f'saved to {out_fp}')
        else:
            out_ds = Dataset.from_dict(data_dict_gathered)
            out_ds.save_to_disk(os.path.join(args.sft_model_path, args.out_name), num_proc=1)

    # accelerator.wait_for_everyone()
    