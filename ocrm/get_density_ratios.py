import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import tyro
from accelerate import Accelerator, PartialState
from accelerate.utils import broadcast, gather_object, tqdm
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from rich.console import Console
from rich.pretty import pprint
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
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
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    local_batch_size: Optional[int] = 32
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""

    # other args
    base_model: str = "models/EleutherAI/pythia-160m-deduped/sft_model_2"
    """the name of the pretrained model to use"""
    sft_model_path: str = "models/EleutherAI/pythia-160m-deduped/sft_model_2"
    """the name of the pretrained model to use"""
    ppo_model_path: str = "models/EleutherAI/pythia-160m-deduped/policy_model_2"
    """the name of the pretrained model to use"""
    dataset_fp: str = "models/EleutherAI/pythia-160m-deduped/sft_model_2/dataset_full_goldadded"
    """the query dataset"""
    response_length: int = 53
    """the length of the response"""
    truncate_token: Literal["eos"] = "eos"
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 1.0
    """the sampling temperature"""
    output_name: str = 'dataset_full_densityratio'


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


def forward(model, query_responses, pad_token_id):
    attention_mask = query_responses != pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=False,
    )


def get_logprobs_sumlogprob(temperature, pad_token_id, eos_token_id, ref_policy, query_response, context_length, mask):
    response1 = query_response[:, context_length:]
    with record_function("model_forward"):
        outputs1 = forward(ref_policy, query_response, pad_token_id)

    with record_function("scaled_softmax"):
        logits1 = outputs1.logits[:, context_length - 1 : -1]
        logits1 /= temperature + 1e-7
        all_logprob1 = F.log_softmax(logits1, dim=-1)

    with record_function("gather"):
        logprob1 = torch.gather(all_logprob1, 2, response1.unsqueeze(-1)).squeeze(-1)

    new_mask = True
    if not new_mask:
        with record_function("mask"):    
            eos_mask = response1 == eos_token_id
        with record_function("cumsum"):    
            eos_indices = eos_mask.cumsum(dim=1)  # this makes everything before the first eos token 0, but the eos token itself is 1, second eos token is 2, etc.
        with record_function("substract_mask"):    
            eos_indices[eos_mask] -= 1  # this makes the first eos token 0
        with record_function("mask_to_float"):    
            mask = (eos_indices == 0).float() 
    
    with record_function("apply_mask"):    
        logprob1_masked = logprob1 * mask
    with record_function("sum_logprob"):    
        sum_logprob1 = logprob1_masked.sum(dim=1)

    return logprob1, sum_logprob1


def get_densities(args, accelerator, dataloader, tokenizer, ref_policy, stop_idx=None):
    with torch.no_grad():
        return_list = []
        
        for data in tqdm(dataloader, desc="Calc Density", total=len(dataloader)):
            query = data["query_token"]
            context_length = query.shape[1]

            queries_decoded = tokenizer.batch_decode(query.cpu().numpy())

            logprob_chosen, sumlogprob_chosen = get_logprobs_sumlogprob(
                args.temperature, tokenizer.pad_token_id, tokenizer.eos_token_id, ref_policy, data['query_chosen_token'], 
                context_length, data['query_chosen_mask'])

            logprob_rejected, sumlogprob_rejected = get_logprobs_sumlogprob(
                args.temperature, tokenizer.pad_token_id, tokenizer.eos_token_id, ref_policy, data['query_rejected_token'], 
                context_length, data['query_rejected_mask'])
            
            with record_function("to_cpu"):
                data_dict = {
                    "queries": queries_decoded,
                    "logprob_chosen": logprob_chosen.float().cpu().numpy(),
                    "sumlogprob_chosen": sumlogprob_chosen.float().cpu().numpy(),
                    "logprob_rejected": logprob_rejected.float().cpu().numpy(),
                    "sumlogprob_rejected": sumlogprob_rejected.float().cpu().numpy(),
                    "gen_id": data['gen_id'].cpu().numpy(),
                }
            return_list.append(data_dict)

            if stop_idx is not None and len(return_list) >= stop_idx:
                break

    accelerator.wait_for_everyone()
    data_dict_gathered = gather_object(return_list)
    return data_dict_gathered


def main():
    args = parse_args()
    accelerator = Accelerator(even_batches=False)
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )

    # load dataset
    use_numpy_load = False
    if use_numpy_load:
        dataset_dict = np.load(args.dataset_fp, allow_pickle=True).item()
        dataset = Dataset.from_dict(dataset_dict)
    else:
        dataset = Dataset.load_from_disk(args.dataset_fp)#, keep_in_memory=True)
        dataset_dict = dataset.data
    columns = [
        "query_token",
        "chosen_token",
        "query_chosen_token",
        "rejected_token",
        "query_rejected_token",
        "query_response1_token",
        "query_response2_token",
        "logprob1",
        "sumlogprob1",
        "logprob2",
        "sumlogprob2",
        'winner',
        'gen_id',
    ]
    dataset = dataset.with_format(
        "torch",
        columns=columns,
    )

    def add_eos_mask(example_batch):
        query_chosen_list = []
        query_rejected_list = []
        for idx in range(len(example_batch['query_token'])):
            context_length = len(example_batch["query_token"][idx])
            query_chosen = np.array(example_batch['query_chosen_token'][idx])

            eos_pos = np.nonzero(query_chosen == tokenizer.eos_token_id)
            if any(eos_pos[0]):
                eos_pos = eos_pos[0][0]
                query_chosen_mask = np.cumsum(query_chosen == tokenizer.eos_token_id)
                query_chosen_mask[eos_pos] = 0
                query_chosen_mask = np.where(query_chosen_mask == 0, 1, 0)
                query_chosen_list.append(torch.tensor(query_chosen_mask[context_length:], dtype=torch.float32))
            else:
                query_chosen_list.append(torch.ones(len(query_chosen) - context_length, dtype=torch.float32))

            query_rejected = np.array(example_batch['query_rejected_token'][idx])
            eos_pos = np.nonzero(query_rejected == tokenizer.eos_token_id)
            if any(eos_pos[0]):
                eos_pos = eos_pos[0][0]
                query_rejected_mask = np.cumsum(query_rejected == tokenizer.eos_token_id)
                query_rejected_mask[eos_pos] = 0
                query_rejected_mask = np.where(query_rejected_mask == 0, 1, 0)
                query_rejected_list.append(torch.tensor(query_rejected_mask[context_length:], dtype=torch.float32))
            else:
                query_rejected_list.append(torch.ones(len(query_rejected) - context_length, dtype=torch.float32))
        example_batch = {k: torch.tensor(v) for k, v in example_batch.items()}
        example_batch.update({
            'query_chosen_mask': torch.stack(query_chosen_list),
            'query_rejected_mask': torch.stack(query_rejected_list)
        })
        return example_batch
    # dataset = dataset.select(range(10000))
    # dataset = dataset.map(add_eos_mask, num_proc=min(len(os.sched_getaffinity(0)) // 2, 64))
    dataset_tf = dataset.with_transform(add_eos_mask, columns=columns)

    dataloader = DataLoader(dataset_tf, batch_size=args.local_batch_size, shuffle=False)
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

    ref_policy = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, 
        config=model_config, 
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    policy = AutoModelForCausalLM.from_pretrained(
        args.ppo_model_path, 
        config=model_config, 
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    for module in [policy, ref_policy]:
        disable_dropout(module)
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    ref_policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    ref_policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

    # policy = torch.compile(policy)

    torch.manual_seed(args.seed)
    dataloader, ref_policy = accelerator.prepare(dataloader, ref_policy)
    torch.manual_seed(local_seed)  # reset the local seed again

    ref_policy.eval()
    data_dict_ref = get_densities(args, accelerator, dataloader, tokenizer, ref_policy)# , stop_idx=25)  # only do 25 for reference model tests
    ref_policy.to('cpu')
    torch.cuda.empty_cache()

    dataloader2 = DataLoader(dataset_tf, batch_size=args.local_batch_size, shuffle=False)
    dataloader2, policy = accelerator.prepare(dataloader2, policy)

    policy.eval()
    data_dict_policy = get_densities(args, accelerator, dataloader2, tokenizer, policy)
    policy.to('cpu')
    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        data_dict_ref = {
            k: np.concatenate([d[k] for d in data_dict_ref], axis=0)
            for k in data_dict_ref[0].keys()
        }
        
        gen_ids = np.array(data_dict_ref['gen_id'])[:len(dataset)]
        index = np.argsort(gen_ids)
        data_dict_ref = {k: np.array(v)[index] for k, v in data_dict_ref.items()}
        matches = [str(q_new) == str(q_ref) for q_new, q_ref in zip(data_dict_ref['queries'], dataset_dict['query'])]
        if not all(matches):
            print(f"queries not matched, mismatched indices: {np.where(~np.array(matches))[0]}")
            if len(np.where(~np.array(matches))[0]) < 10:
                print('mismatched queries:')
                for idx in np.where(~np.array(matches))[0]:
                    print(f'###########new {idx}: {data_dict_ref["queries"][idx]}')
                    print(f'###########ref {idx}: {dataset_dict["query"][idx]}')
            else:
                raise ValueError('queries not matched')
        
        data_len = len(data_dict_ref['queries'])  # needed because we don't do the full dataset for reference model tests checking

        dataset_chosen_sumlogprob = np.where(dataset_dict['winner'], dataset_dict['sumlogprob2'], dataset_dict['sumlogprob1'])
        dataset_rejected_sumlogprob = np.where(dataset_dict['winner'], dataset_dict['sumlogprob1'], dataset_dict['sumlogprob2'])
        chosen_diff = np.abs(data_dict_ref['sumlogprob_chosen'] - dataset_chosen_sumlogprob[:data_len])
        rejected_diff = np.abs(data_dict_ref['sumlogprob_rejected'] - dataset_rejected_sumlogprob[:data_len])
        print('mean chosen diff:', np.mean(chosen_diff))
        print('mean rejected diff:', np.mean(rejected_diff))

        data_dict_policy = {
            k: np.concatenate([d[k] for d in data_dict_policy], axis=0)
            for k in data_dict_policy[0].keys()
        }
        gen_ids = np.array(data_dict_policy['gen_id'])[:len(dataset)]
        index = np.argsort(gen_ids)
        data_dict_policy = {k: np.array(v)[index] for k, v in data_dict_policy.items()}
        matches = [str(q_new) == str(q_ref) for q_new, q_ref in zip(data_dict_policy['queries'], dataset_dict['query'])]
        if not all(matches):
            print(f"queries not matched, mismatched indices: {np.where(~np.array(matches))[0]}")
            if len(np.where(~np.array(matches))[0]) < 10:
                print('mismatched queries:')
                for idx in np.where(~np.array(matches))[0]:
                    print(f'new: {data_dict_policy["queries"][idx]}')
                    print(f'ref: {dataset_dict["query"][idx]}')
            else:
                raise ValueError('queries not matched')

        if len(data_dict_ref['queries']) != len(data_dict_policy['queries']):
            data_len = len(data_dict_policy['queries'])
            log_density_ratio_chosen = data_dict_policy['sumlogprob_chosen'] - dataset_chosen_sumlogprob[:data_len]
            log_density_ratio_rejected = data_dict_policy['sumlogprob_rejected'] - dataset_rejected_sumlogprob[:data_len]
        else:
            log_density_ratio_chosen = data_dict_policy['sumlogprob_chosen'] - data_dict_ref['sumlogprob_chosen']
            log_density_ratio_rejected = data_dict_policy['sumlogprob_rejected'] - data_dict_ref['sumlogprob_rejected']
        
        print('Mean log density ratio chosen:', np.mean(log_density_ratio_chosen))
        print('Mean log density ratio rejected:', np.mean(log_density_ratio_rejected))
        print('Mean density ratio chosen:', np.mean(np.exp(log_density_ratio_chosen)))
        print('Mean density ratio rejected:', np.mean(np.exp(log_density_ratio_rejected)))

        plt.hist(log_density_ratio_chosen, label='chosen', fill=False, histtype='step', bins=20)
        plt.hist(log_density_ratio_rejected, label='rejected', fill=False, histtype='step', bins=20)
        plt.legend()
        plt.xlabel('log density ratio')
        plt.ylabel('count')
        plot_fp = os.path.join(args.sft_model_path, 'density_ratio_hist.png')
        plt.savefig(plot_fp)
        plt.close()
        print(f'plotted to {plot_fp}')

        use_numpy_save = False
        if use_numpy_save:
            dataset_dict['log_density_ratio_chosen'] = log_density_ratio_chosen
            dataset_dict['log_density_ratio_rejected'] = log_density_ratio_rejected

            dataset_dict['logprobs_chosen_policy'] = data_dict_policy['logprob_chosen']
            dataset_dict['logprobs_rejected_policy'] = data_dict_policy['logprob_rejected']
            dataset_dict['sumlogprobs_chosen_policy'] = data_dict_policy['sumlogprob_chosen']
            dataset_dict['sumlogprobs_rejected_policy'] = data_dict_policy['sumlogprob_rejected']
            out_fp = os.path.join(args.sft_model_path, args.output_name + '.npy')
            np.save(out_fp, dataset_dict)
            print(f'saved to {out_fp}')
        else:
            dataset = dataset.add_column('log_density_ratio_chosen', log_density_ratio_chosen)
            dataset = dataset.add_column('log_density_ratio_rejected', log_density_ratio_rejected)
            dataset = dataset.add_column('logprobs_chosen_policy', list(data_dict_policy['logprob_chosen']))
            dataset = dataset.add_column('logprobs_rejected_policy', list(data_dict_policy['logprob_rejected']))
            dataset = dataset.add_column('sumlogprobs_chosen_policy', list(data_dict_policy['sumlogprob_chosen']))
            dataset = dataset.add_column('sumlogprobs_rejected_policy', list(data_dict_policy['sumlogprob_rejected']))
            dataset = dataset.add_column('logprobs_chosen_ref', list(data_dict_ref['logprob_chosen']))
            dataset = dataset.add_column('logprobs_rejected_ref', list(data_dict_ref['logprob_rejected']))
            dataset = dataset.add_column('sumlogprobs_chosen_ref', list(data_dict_ref['sumlogprob_chosen']))
            dataset = dataset.add_column('sumlogprobs_rejected_ref', list(data_dict_ref['sumlogprob_rejected']))
            out_fp = os.path.join(args.sft_model_path, args.output_name)
            dataset.save_to_disk(out_fp, num_proc=1)
            print(f'saved to {out_fp}')

        # param_diff = torch.mean(torch.stack([torch.mean(torch.abs(p1 - p2)) for p1, p2 in zip(policy.parameters(), ref_policy.parameters())]))
        # print('Mean parameter difference:', param_diff)


if __name__ == "__main__":
    main()
