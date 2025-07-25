import os
import sys
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional
import copy
import multiprocessing
import pdb

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast
from datasets import Dataset, load_dataset, load_from_disk
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    get_scheduler,
)
from huggingface_hub import HfApi

from ocrm.gold_label_dataset import get_reward_scores as get_gold_reward_scores

api = HfApi()
INVALID_LOGPROB = 1.0


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 50
    """How often to print sample output"""
    run_eval: bool = False
    """Whether to run evaluation"""
    run_gold_eval: bool = True
    """Whether to load and run the gold evaluation at every print_sample_output_freq"""

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
    gradient_accumulation_steps: int = 8
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 16
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = 92832
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_eval_batch_size: int = 16
    """per rank eval batch size"""
    local_rollout_forward_batch_size: int = 4 #64
    """per rank rollout forward batch size, only used in evaluation"""

    # other args
    base_model: str = "EleutherAI/pythia-160m"
    """the name of the pretrained model to use"""
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    """the query dataset"""
    eval_dataset: str = "vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144"
    """ eval ds """
    sft_model_path: str = "models/EleutherAI/pythia-410m-deduped/sft_model_1"
    """the path to the sft model"""
    response_length: int = 53
    """the length of the response"""
    truncate_token: Literal["eos"] = "eos"
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    train_dataset: str = "vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
    ipo: bool = False
    """Whether to use IPO loss https://arxiv.org/abs/2310.12036"""
    label_smoothing: float = 0.0
    """Label smoothing for DPO (Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf))"""
    beta: float = 0.05
    """The beta value for DPO"""

    # wandb and HF tracking configs
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    push_to_hub: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """the user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """the id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """the revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """the url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "models/dpo_model"
    """Where to save the model"""


def parse_args() -> tuple[Args, Accelerator]:
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    if args.push_to_hub:
        if args.hf_repo_id is None: # auto-generate one
            args.hf_repo_id = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        if args.hf_entity is None:  # find the current user
            args.hf_entity = api.whoami()["name"]
        if "/" not in args.hf_repo_id: # prepend the current user
            args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    return args, accelerator


# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def forward(model, query_responses, labels, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    labels = labels[:, 1:].clone()
    logits = output.logits[:, :-1, :]
    loss_mask = (labels != tokenizer.pad_token_id)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    all_logps = (per_token_logps * loss_mask).sum(-1)
    chosen_logps = all_logps[:query_responses.shape[0] // 2]
    rejected_logps = all_logps[query_responses.shape[0] // 2:]
    return chosen_logps, rejected_logps

def forward_single(model, query_responses, labels, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    labels = labels[:, 1:].clone()
    logits = output.logits[:, :-1, :]
    loss_mask = (labels != tokenizer.pad_token_id)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    logps_masked = (per_token_logps * loss_mask)
    return logps_masked

def forward_simple(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def generate_for_kl(lm_backbone, queries, tokenizer, generation_config):
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
        output_scores=True
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits

@torch.no_grad()
def get_validation_kl_gen(kl_validation_ref_policy, policy, dataloader, generation_config, n_batches, args, tokenizer, accelerator):
    kl_validation_ref_policy.to(accelerator.device)
    kl_divs = []
    if n_batches == 0:
        n_batches = len(dataloader)
    for batch_idx, data in tqdm(enumerate(dataloader), desc='KL Validation', total=n_batches, disable=n_batches > 16):
        queries = data["query_token"].to(accelerator.device)
        context_length = queries.shape[1]
        query_responses = []
        responses = []
        postprocessed_responses = []
        logprobs = []
        ref_logprobs = []
        values = []
        sequence_lengths = []

        for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
            query = queries[i : i + args.local_rollout_forward_batch_size]
            query_response, logits = generate_for_kl(
                policy,
                query,
                tokenizer,
                generation_config,
            )
            response = query_response[:, context_length:]

            # use the logits during generation directly, instead of using the following
            all_logprob = F.log_softmax(logits, dim=-1)
            logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
            # del logits, all_logprob
            torch.cuda.empty_cache()

            ref_output = forward_simple(kl_validation_ref_policy, query_response, tokenizer)
            ref_logits = ref_output.logits[:, context_length - 1 : -1]
            ref_logits /= args.temperature + 1e-7
            ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
            ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
            # del ref_output, ref_logits, ref_all_logprob
            torch.cuda.empty_cache()

            # Response Processing 1. truncate response after the first occurrence of `truncate_token_id`
            postprocessed_response = truncate_response(args, tokenizer, response)

            # Response Processing 2. run reward model on the truncated responses    
            sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1

            query_responses.append(query_response)
            responses.append(response)
            postprocessed_responses.append(postprocessed_response)
            logprobs.append(logprob)
            ref_logprobs.append(ref_logprob)
            sequence_lengths.append(sequence_length)
        query_responses = torch.cat(query_responses, 0)
        responses = torch.cat(responses, 0)
        postprocessed_responses = torch.cat(postprocessed_responses, 0)
        logprobs = torch.cat(logprobs, 0)
        ref_logprobs = torch.cat(ref_logprobs, 0)
        sequence_lengths = torch.cat(sequence_lengths, 0)
        del logprob, ref_logprob
        torch.cuda.empty_cache()


        # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

        # 4. compute rewards
        kl = logprobs - ref_logprobs
        kl_divs.append(kl.sum(1).cpu().numpy())

        if n_batches > 0 and batch_idx == n_batches - 1:
            break
    
    torch.cuda.empty_cache()
    return np.mean(kl_divs)

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
    )
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)


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


def run_gold_eval(eval_df, gold_rm, gold_rm_tokenizer, accelerator):
    gold_rm.to(accelerator.device)
    eval_df_gilded = eval_df.copy()
    model_scores = []
    ref_scores = []
    for i, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc='Gold Eval'):
        prompt = row["query"]
        model_response = row["postprocessed_response"]
        ref_response = row["reference_responses"].replace('<|endoftext|>','').replace('[PAD]','')
        score_model, score_ref = get_gold_reward_scores(gold_rm, gold_rm_tokenizer, [prompt], [model_response], [ref_response], accelerator.device)
        model_scores.append(score_model.item())
        ref_scores.append(score_ref.item())
    eval_df_gilded["gold_model_scores"] = model_scores
    eval_df_gilded["gold_ref_scores"] = ref_scores
    gold_rm.to("cpu")
    torch.cuda.empty_cache()
    mean_gold_model_score = np.mean(model_scores)
    mean_gold_ref_score = np.mean(ref_scores)
    model_winrate = np.mean(np.array(model_scores) > np.array(ref_scores))
    return eval_df_gilded, mean_gold_model_score, mean_gold_ref_score, model_winrate


def evaluate_rm(args: Args, accelerator, tokenizer, model, ref_model, dataloader):
    model.eval()
    with torch.no_grad():
        items = defaultdict(list)
        for data in tqdm(dataloader):
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            labels = torch.cat((data["query_chosen_token_response_label"], data["query_rejected_token_response_label"]), dim=0)
            ref_chosen_logps, ref_rejected_logps = forward(ref_model, query_responses, labels, tokenizer)
            chosen_logps, rejected_logps = forward(model, query_responses, labels, tokenizer)
            reward_preferred = args.beta * (chosen_logps - ref_chosen_logps)
            reward_rejected = args.beta * (rejected_logps - ref_rejected_logps)
            accuracy = reward_preferred > reward_rejected
            accuracy = accelerator.gather(accuracy)
            reward_preferred = accelerator.gather(reward_preferred)
            reward_rejected = accelerator.gather(reward_rejected)
            for k in data:
                data[k] = gather_object(data[k])
            for i in range(len(accuracy)):
                items["query"].append(tokenizer.decode(data["query_token"][i], skip_special_tokens=True))
                items["chosen"].append(tokenizer.decode(data["chosen_token"][i]))
                items["rejected"].append(tokenizer.decode(data["rejected_token"][i]))
                items["batch"].append(data["batch"][i])
                items["split"].append(data["split"][i])
                items["confidence"].append(data["extra.confidence"][i].item())
                items["choice"].append(data["choice"][i].item())
                items["policies"].append(data["policies"][i])
                items["chosen_policy"].append(data["chosen_policy"][i])
                items["rejected_policy"].append(data["rejected_policy"][i])
                items["accuracy"].append(accuracy[i].item())
                items["reward_preferred"].append(reward_preferred[i].item())
                items["reward_rejected"].append(reward_rejected[i].item())
    model.train()
    return pd.DataFrame(items)



@dataclass
class EvalStorage:
    query_token: List[str] = field(default_factory=list)
    postprocessed_response_token: List[str] = field(default_factory=list)
    reference_response_token: List[str] = field(default_factory=list)
    score: List[float] = field(default_factory=list)
    reference_score: List[float] = field(default_factory=list)

    query: List[str] = field(default_factory=list)
    postprocessed_response: List[str] = field(default_factory=list)
    reference_response: List[str] = field(default_factory=list)


def evaluate_policy(args: Args, model, tokenizer, dataloader, generation_config, n_batches=-1):
    eval_storage = EvalStorage()
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(dataloader), desc='eval', disable=n_batches > 0):
            queries = data["query_token"]
            reference_response_token = data["reference_response_token"]
            context_length = queries.shape[1]
            query_responses = generate(
                model,
                queries,
                tokenizer,
                generation_config,
            )
            responses = query_responses[:, context_length:]
            postprocessed_responses = truncate_response(args, tokenizer, responses)
            eval_storage.query_token.extend(queries)
            eval_storage.reference_response_token.extend(reference_response_token)
            eval_storage.postprocessed_response_token.extend(postprocessed_responses)
            if n_batches > 0 and batch_idx == n_batches - 1:
                break

    eval_storage.query = tokenizer.batch_decode(eval_storage.query_token, skip_special_tokens=True)
    eval_storage.reference_response = tokenizer.batch_decode(eval_storage.reference_response_token)
    eval_storage.postprocessed_response = tokenizer.batch_decode(
        eval_storage.postprocessed_response_token, skip_special_tokens=True
    )
    # eval_score = torch.cat(eval_storage.score).float().cpu().numpy().tolist()
    # eval_reference_score = torch.cat(eval_storage.reference_score).float().cpu().numpy().tolist()
    eval_df = pd.DataFrame(
        {
            "query": gather_object(eval_storage.query),
            "postprocessed_response": gather_object(eval_storage.postprocessed_response),
            "reference_responses": gather_object(eval_storage.reference_response),
            # "scores": gather_object(eval_score),
            # "reference_scores": gather_object(eval_reference_score),
        }
    )
    return eval_storage, eval_df

@torch.no_grad()
def get_dpo_validation_kl(model, ref_model, dataloader, n_batches, args, tokenizer):
    kl_divs = []
    if n_batches == 0:
        n_batches = len(dataloader)
    for batch_idx, data in tqdm(enumerate(dataloader), desc='KL Validation', total=n_batches):
        query_responses = data["query_reference_response_token"]
        labels = data["query_reference_response_token_response_label"]

        logps = forward_single(model, query_responses, labels, tokenizer)
        ref_logps = forward_single(ref_model, query_responses, labels, tokenizer)

        kl = ref_logps - logps
        kl_divs.append(kl.sum(1).cpu().numpy())

        if n_batches > 0 and batch_idx == n_batches - 1:
            break
    
    torch.cuda.empty_cache()
    return np.mean(kl_divs)


# def train(args: Args):
if __name__ == "__main__":
    args, accelerator = parse_args()
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if args.truncate_token == "eos":
        args.truncate_token_id = tokenizer.eos_token_id

    # load dataset
    if not 'vwxyzjn' in args.train_dataset:
        dataset = load_from_disk(args.train_dataset)
        print('processing DS...')
        dataset = dataset.shuffle(seed=local_seed)
        if not args.total_episodes == -1:
            if args.total_episodes > len(dataset):
                args.total_episodes = len(dataset)
            dataset = dataset.select(range(args.total_episodes))

        def process_data(x):
            unpadded_query_token = [token for token in x["query_token"] if token != tokenizer.pad_token_id]
            # pdb.set_trace()
            x["query_chosen_token_response_label"] = np.array(x["query_chosen_token"])
            x["query_chosen_token_response_label"][:len(unpadded_query_token)] = [tokenizer.pad_token_id for _ in range(len(unpadded_query_token))]
            x["query_rejected_token_response_label"] = np.array(x["query_rejected_token"])
            x["query_rejected_token_response_label"][:len(unpadded_query_token)] = [tokenizer.pad_token_id for _ in range(len(unpadded_query_token))]
            return x
        dataset = dataset.map(process_data, load_from_cache_file=False, 
                        # num_proc =8)
                        num_proc=multiprocessing.cpu_count())
        print('processed ds')
    else:
        dataset = load_dataset(args.train_dataset, split="train")
        dataset = dataset.shuffle(seed=local_seed)
        if not args.total_episodes == -1:
            if args.total_episodes > len(dataset):
                args.total_episodes = len(dataset)
            dataset = dataset.select(range(args.total_episodes))
    dataset = dataset.with_format(
        "torch",
        columns=[
            "query_token",
            "chosen_token",
            "query_chosen_token",
            "query_chosen_token_response_label",
            "rejected_token",
            "query_rejected_token",
            "query_rejected_token_response_label",
        ],
    )
    dataloader = DataLoader(dataset, batch_size=args.local_micro_batch_size)
    eval_datasets = []
    eval_dataloaders = {}
    for split in ["validation", "validation_cnndm"]:
        validation_dataset = load_dataset(args.eval_dataset, split=split).flatten()
        validation_dataset = validation_dataset.with_format(
            "torch",
            columns=[
                "query_token",
                "choice",
                "chosen_token",
                "query_chosen_token",
                "query_chosen_token_response_label",
                "rejected_token",
                "query_rejected_token",
                "query_rejected_token_response_label",
                "batch",
                "split",
                "extra.confidence",
                "chosen_policy",
                "rejected_policy",
                "policies",
            ],
        )
        eval_datasets.append(validation_dataset)
        eval_dataloaders[split] = DataLoader(validation_dataset, batch_size=args.local_eval_batch_size)
    sft_validation_dataset = load_dataset(args.query_dataset, split="validation")
    sft_validation_dataset = sft_validation_dataset.with_format("torch", columns=["query_token", "query_reference_response_token", "reference_response_token", "query_reference_response_token_response_label"])
    sft_validation_dataloader = DataLoader(sft_validation_dataset, batch_size=args.local_eval_batch_size)

    args.total_episodes = len(dataset)
    args.num_updates = args.total_episodes // args.batch_size

    
    if args.run_gold_eval:
        gold_rm_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
        gold_rm = AutoModelForSequenceClassification.from_pretrained(
            gold_rm_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        gold_rm.eval()
        gold_rm_tokenizer = AutoTokenizer.from_pretrained(gold_rm_name)
    
    console = Console(force_terminal=True)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    if accelerator.is_main_process:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=args.run_name,
                save_code=True,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]) and not '.venv' in path)
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True
    model_config = AutoConfig.from_pretrained(args.sft_model_path)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        config=model_config,
        trust_remote_code=True,
    )
    ref_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        config=model_config,
        trust_remote_code=True,
    )
    disable_dropout(model)
    model.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    model.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_updates * args.num_train_epochs,
    )

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    eval_dataloaders = {split: accelerator.prepare(eval_dataloader) for split, eval_dataloader in eval_dataloaders.items()}
    sft_validation_dataloader = accelerator.prepare(sft_validation_dataloader)
    torch.manual_seed(local_seed)  # reset the local seed again

    ref_model = ref_model.to(device)
    # use the same `0.01` temperature for validation response generation https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/exps/sample.py#L27
    validation_generation_config = GenerationConfig(
        max_new_tokens=128,
        min_new_tokens=128,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        min_new_tokens=args.response_length,
        temperature=(args.temperature + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    accelerator.print("===training model===")
    losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
    accuracies = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_preferreds = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_rejecteds = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_margins = torch.zeros((args.gradient_accumulation_steps,), device=device)
    model.train()
    gradient_accumulation_idx = 0
    global_step = 0
    update = 0
    for epoch in range(args.num_train_epochs):
        accelerator.print(f"epoch: {epoch}")
        for data in tqdm(dataloader, desc='training steps', dynamic_ncols=True):
            if update % args.print_sample_output_freq == 0:
                _, evaluate_df = evaluate_policy(
                    args, 
                    accelerator.unwrap_model(model), 
                    tokenizer, 
                    sft_validation_dataloader, 
                    validation_generation_config, 
                    n_batches = 16 // accelerator.num_processes
                )
                if args.run_gold_eval:
                    torch.cuda.empty_cache()
                    eval_df_gilded, eval_model_gold_score, eval_ref_gold_score, eval_model_winrate = \
                        run_gold_eval(evaluate_df, gold_rm, gold_rm_tokenizer, accelerator)
                    evaluate_df = eval_df_gilded
                    writer.add_scalar("eval/gold_model_score", eval_model_gold_score, global_step)
                    writer.add_scalar("eval/gold_ref_score", eval_ref_gold_score, global_step)
                    writer.add_scalar("eval/gold_model_winrate", eval_model_winrate, global_step)
                    accelerator.print(f"{eval_model_gold_score=:.2f}, {eval_ref_gold_score=:.2f}, {eval_model_winrate=:.2f}")
                    validation_kl_dpo = get_dpo_validation_kl(
                        accelerator.unwrap_model(model),
                        ref_model,
                        sft_validation_dataloader,
                        n_batches = 16 // accelerator.num_processes,
                        args=args,
                        tokenizer=tokenizer,
                    )
                    writer.add_scalar("eval/validation_kl_dpo", validation_kl_dpo, global_step)
                    accelerator.print(f"{validation_kl_dpo=:.3f}")
                    
                    validation_kl_gen = get_validation_kl_gen(
                        ref_model,
                        accelerator.unwrap_model(model),
                        sft_validation_dataloader,
                        generation_config,
                        n_batches = 16 // accelerator.num_processes,
                        args=args,
                        tokenizer=tokenizer,
                        accelerator=accelerator
                    )
                    writer.add_scalar("eval/validation_kl", validation_kl_gen, global_step)
                    accelerator.print(f"{validation_kl_gen=:.3f}")

                    if accelerator.is_main_process:
                        eval_split = 'validation'
                        eval_idx = 1
                        eval_ds = Dataset.from_pandas(evaluate_df)
                        eval_ds.save_to_disk(f"runs/{args.run_name}/{eval_split}_dataset_{global_step}_nr{eval_idx}")
                        if args.track:
                            wandb.log({f"samples/{eval_split}_query_responses_nr{eval_idx}": wandb.Table(dataframe=evaluate_df)}, step=update)

            update += 1
            global_step += args.micro_batch_size
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            labels = torch.cat((data["query_chosen_token_response_label"], data["query_rejected_token_response_label"]), dim=0)
            with torch.no_grad():
                ref_chosen_logps, ref_rejected_logps = forward(ref_model, query_responses, labels, tokenizer)
            with accelerator.accumulate(model):
                chosen_logps, rejected_logps = forward(model, query_responses, labels, tokenizer)
                pi_logratios = chosen_logps - rejected_logps
                ref_logratios = ref_chosen_logps - ref_rejected_logps
                logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}
                if args.ipo:
                    loss = (logits - 1/(2 * args.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
                else:
                    loss = -F.logsigmoid(args.beta * logits) * (1 - args.label_smoothing) - F.logsigmoid(-args.beta * logits) * args.label_smoothing
                loss = torch.mean(loss)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            with torch.no_grad():
                reward_preferred = args.beta * (chosen_logps - ref_chosen_logps)
                reward_rejected = args.beta * (rejected_logps - ref_rejected_logps)
                losses[gradient_accumulation_idx] = loss
                accuracies[gradient_accumulation_idx] = (reward_preferred > reward_rejected).float().mean()
                reward_preferreds[gradient_accumulation_idx] = reward_preferred.mean()
                reward_rejecteds[gradient_accumulation_idx] = reward_rejected.mean()
                reward_margins[gradient_accumulation_idx] = (reward_preferred - reward_rejected).mean()
            gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
            if update > 1 and (update - 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()
                writer.add_scalar("train/pi_logratio", accelerator.gather(pi_logratios).mean().item(), global_step)
                writer.add_scalar("train/ref_logratio", accelerator.gather(ref_logratios).mean().item(), global_step)
                writer.add_scalar("train/logits", accelerator.gather(logits).mean().item(), global_step)
                train_accuracy = accelerator.gather(accuracies).mean().item()
                writer.add_scalar("train/rm/loss", accelerator.gather(losses).mean().item(), global_step)
                writer.add_scalar("train/rm/accuracy", train_accuracy, global_step)
                writer.add_scalar(
                    "train/rm/reward_preferred", accelerator.gather(reward_preferreds).mean().item(), global_step
                )
                writer.add_scalar("train/rm/reward_rejected", accelerator.gather(reward_rejecteds).mean().item(), global_step)
                writer.add_scalar("train/rm/reward_margin", accelerator.gather(reward_margins).mean().item(), global_step)
                writer.add_scalar("train/rm/lr", scheduler.get_last_lr()[0], global_step)
                accelerator.print(
                    f"{train_accuracy=}, {scheduler.get_last_lr()=}, {optimizer.param_groups[0]['lr']=}, {update=}"
                )

    if args.run_eval:
        _, evaluate_df = evaluate_policy(args, model, tokenizer, sft_validation_dataloader, validation_generation_config)
        if args.run_gold_eval:
            torch.cuda.empty_cache()
            eval_df_gilded, eval_model_gold_score, eval_ref_gold_score, eval_model_winrate = \
                run_gold_eval(evaluate_df, gold_rm, gold_rm_tokenizer, accelerator)
            evaluate_df = eval_df_gilded
            writer.add_scalar("eval/gold_model_score", eval_model_gold_score, global_step)
            writer.add_scalar("eval/gold_ref_score", eval_ref_gold_score, global_step)
            writer.add_scalar("eval/gold_model_winrate", eval_model_winrate, global_step)
            accelerator.print(f"{eval_model_gold_score=:.2f}, {eval_ref_gold_score=:.2f}, {eval_model_winrate=:.2f}")

        if accelerator.is_main_process:
            evaluate_df.to_csv(f"runs/{args.run_name}/table.csv")
            if args.track:
                wandb.log({"eval/query_responses": wandb.Table(dataframe=evaluate_df)}, step=update)
        del evaluate_df
        torch.cuda.empty_cache()
        # for eval_split in eval_dataloaders:
        #     evaluate_df = evaluate_rm(args, accelerator, tokenizer, model, ref_model, eval_dataloaders[eval_split])
        #     for split, row in evaluate_df[["split", "accuracy"]].groupby(["split"]).mean().iterrows():
        #         writer.add_scalar(f"eval/rm/{eval_split}/accuracy/split/{split}", row["accuracy"], global_step)
        #         accelerator.print(f"eval/rm/{eval_split}/accuracy/split/{split}: {row['accuracy']}")
        #     for batch, row in evaluate_df[["batch", "accuracy"]].groupby(["batch"]).mean().iterrows():
        #         writer.add_scalar(f"eval/rm/{eval_split}/accuracy/batch/{batch}", row["accuracy"], global_step)
        #         accelerator.print(f"eval/rm/{eval_split}/accuracy/batch/{batch}: {row['accuracy']}")
        #     for confi, row in evaluate_df[["confidence", "accuracy"]].groupby(["confidence"]).mean().iterrows():
        #         writer.add_scalar(f"eval/rm/{eval_split}/accuracy/confidence/{confi}", row["accuracy"], global_step)
        #         accelerator.print(f"eval/rm/{eval_split}/accuracy/confidence/{confi}: {row['accuracy']}")
        #     writer.add_scalar(f"eval/rm/{eval_split}/accuracy", evaluate_df["accuracy"].mean(), global_step)
        #     accelerator.print(f"eval/rm/{eval_split}/accuracy: {evaluate_df['accuracy'].mean()}")
        #     if accelerator.is_main_process:
        #         os.makedirs(f"eval_tables/{args.run_name}", exist_ok=True)
        #         evaluate_df.to_csv(f"eval_tables/{args.run_name}/eval_{eval_split}_{update}.csv")
        #         if args.track:
        #             wandb.log({f"samples/{eval_split}/query_responses": wandb.Table(dataframe=evaluate_df)}, step=update)
        #     del evaluate_df
        # torch.cuda.empty_cache()

    # save model
    if args.output_dir and args.num_train_epochs > 0:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        time_tensor = torch.tensor([int(time.time())], device=device)
        time_int = accelerator.gather(time_tensor)[0].item()  # avoid different timestamps across processes
        repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                tokenizer.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision)
        unwrapped: PreTrainedModel = accelerator.unwrap_model(model)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=False,
            )
            if args.push_to_hub:
                unwrapped.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision, safe_serialization=False)
                accelerator.print(f"🔥 pushed to {args.hf_repo_url}")
