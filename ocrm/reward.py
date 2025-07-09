import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast, tqdm
from datasets import load_dataset, Dataset, load_from_disk
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    get_scheduler,
)
from huggingface_hub import HfApi
import wandb

api = HfApi()


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 2
    """seed of the experiment"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 220
    """How often to print sample output"""
    run_eval: bool = True
    """Whether to run evaluation"""

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

    log_density_ratio_mult: float = 0.01
    """The multiplier for the log density ratio, with 0.0 meaning no importance weighting and 1.0 meaning full importance weighting"""
    relative_density_factor: float = 0.1
    """Factor in Relative Density-Ratio Estimation. 0.0 means default importance weights """
    batch_normalize_importance_weights: bool = False
    """Whether to normalize the importance weights across the batch"""
    dataset_normalize_importance_weights: bool = False
    """Whether to normalize the importance weights across the dataset"""
    truncated_iw: int = -1
    """The number of tokens to truncate the importance weights to. -1 means no truncation"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    gradient_accumulation_steps: int = 1
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 16
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = 3*92832
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_eval_batch_size: int = 16
    """per rank eval batch size"""

    # other args
    base_model: str = "EleutherAI/pythia-1b"
    """the name of the pretrained model to use"""
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    """the query dataset"""
    reward_model_path: str = ""
    """the path to the reward model"""
    sft_model_path: str = "models/EleutherAI/pythia-1b-deduped/sft_model_150"
    """the path to the sft model"""
    label_dataset: str = "vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
    custom_df_path: str = "models/EleutherAI/pythia-1b-deduped/sft_model_150/dataset_full_densityratio"
    """training dataset path (might include importance weights)"""
    iw_val_ds_path: Optional[str] = None
    """importance weighting validation data"""

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
    output_dir: str = ""
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


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def get_reward(model, query_responses, tokenizer, context_length=0):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1)

def evaluate_iwds_accuracy(args: Args, accelerator, tokenizer, model, dataloader):
    model.eval()
    with torch.no_grad():
        total_accuracy = []
        total_chosen_rewards = []
        total_rejected_rewards = []
        total_loss = []
        for data in tqdm(dataloader, desc='evaluation', dynamic_ncols=True):
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            with accelerator.accumulate(model):
                predicted_reward = get_reward(model, query_responses, tokenizer)
                chosen_rewards = predicted_reward[:data['query_chosen_token'].shape[0]]
                rejected_rewards = predicted_reward[data['query_chosen_token'].shape[0]:]
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
                accuracy = (chosen_rewards > rejected_rewards).float()
                accuracy = accelerator.gather(accuracy)
                chosen_rewards = accelerator.gather(chosen_rewards)
                rejected_rewards = accelerator.gather(rejected_rewards)
                total_accuracy.extend(accuracy.float().cpu().numpy())
                total_loss.extend(loss.float().cpu().numpy())
                total_chosen_rewards.extend(chosen_rewards.float().cpu().numpy())
                total_rejected_rewards.extend(rejected_rewards.float().cpu().numpy())
    model.train()
    return np.mean(total_accuracy), np.mean(total_chosen_rewards), np.mean(total_rejected_rewards), np.mean(total_loss)


def evaluate(args: Args, accelerator, tokenizer, model, dataloader):
    model.eval()
    with torch.no_grad():
        items = defaultdict(list)
        for data in tqdm(dataloader, desc='evaluation', dynamic_ncols=True):
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            with accelerator.accumulate(model):
                predicted_reward = get_reward(model, query_responses, tokenizer)
                chosen_rewards = predicted_reward[:data['query_chosen_token'].shape[0]]
                rejected_rewards = predicted_reward[data['query_chosen_token'].shape[0]:]
                accuracy = (chosen_rewards > rejected_rewards).float()
                accuracy = accelerator.gather(accuracy)
                chosen_rewards = accelerator.gather(chosen_rewards)
                rejected_rewards = accelerator.gather(rejected_rewards)
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
                items["chosen_rewards"].append(chosen_rewards[i].item())
                items["rejected_rewards"].append(rejected_rewards[i].item())
    model.train()
    return pd.DataFrame(items)


# def train(args: Args):
if __name__ == "__main__":
    args, accelerator = parse_args()
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    # load dataset
    use_custom_ds = True
    importance_weighted = False

    if use_custom_ds:
        use_numpy_ds = False
        if use_numpy_ds:
            dataset_dict = np.load(args.custom_df_path, allow_pickle=True).item()
            if 'log_density_ratio_chosen' in dataset_dict.keys():
                importance_weighted = True
            dataset = Dataset.from_dict(dataset_dict)
        else:
            dataset = load_from_disk(args.custom_df_path)
            if 'log_density_ratio_chosen' in dataset.features:
                importance_weighted = True
        dataset = dataset.shuffle(seed=local_seed)
        dataset = dataset.select(range(min(len(dataset), args.total_episodes)))
        if importance_weighted:
            columns = [
                    "query_token",
                    "chosen_token",
                    "query_chosen_token",
                    "rejected_token",
                    "query_rejected_token",
                    "log_density_ratio_chosen",
                    "log_density_ratio_rejected",
                    "sumlogprobs_chosen_policy",
                    "sumlogprobs_rejected_policy",
                    "sumlogprobs_chosen_ref",
                    "sumlogprobs_rejected_ref",
                    "logprobs_chosen_policy",
                    "logprobs_rejected_policy",
                    "logprobs_chosen_ref",
                    "logprobs_rejected_ref",
                ]
            dataset = dataset.with_format(
                "torch",
                columns=columns,
            )
            raw_log_importance_weights = dataset["log_density_ratio_chosen"] + dataset["log_density_ratio_rejected"]
            if args.truncated_iw > 0:
                print('truncating importance weights')
                sumlogprobs_chosen_policy = dataset["logprobs_chosen_policy"][:, :args.truncated_iw].sum(1)
                sumlogprobs_rejected_policy = dataset["logprobs_rejected_policy"][:, :args.truncated_iw].sum(1)
                sumlogprobs_chosen_ref = dataset["logprobs_chosen_ref"][:, :args.truncated_iw].sum(1)
                sumlogprobs_rejected_ref = dataset["logprobs_rejected_ref"][:, :args.truncated_iw].sum(1)
                
                def truncated_iw_transform(example_batch):
                    example_batch = {k: torch.tensor(v) for k, v in example_batch.items()}
                    example_batch["sumlogprobs_chosen_policy"] = example_batch["logprobs_chosen_policy"][:, :args.truncated_iw].sum(1)
                    example_batch["sumlogprobs_rejected_policy"] = example_batch["logprobs_rejected_policy"][:, :args.truncated_iw].sum(1)
                    example_batch["sumlogprobs_chosen_ref"] = example_batch["logprobs_chosen_ref"][:, :args.truncated_iw].sum(1)
                    example_batch["sumlogprobs_rejected_ref"] = example_batch["logprobs_rejected_ref"][:, :args.truncated_iw].sum(1)
                    return example_batch
                dataset = dataset.with_transform(truncated_iw_transform, columns=columns)
            else:
                sumlogprobs_chosen_policy = dataset["sumlogprobs_chosen_policy"]
                sumlogprobs_rejected_policy = dataset["sumlogprobs_rejected_policy"]
                sumlogprobs_chosen_ref = dataset["sumlogprobs_chosen_ref"]
                sumlogprobs_rejected_ref = dataset["sumlogprobs_rejected_ref"]


            # # or as a dataset transformation
            print('done truncating importance weights')
            if args.relative_density_factor > 0.0:
                alpha = args.relative_density_factor
                importance_chosen = sumlogprobs_chosen_policy - torch.log(
                    alpha * sumlogprobs_chosen_policy.exp() + (1 - alpha) * sumlogprobs_chosen_ref.exp() + 1e-36)
                importance_rejected = sumlogprobs_rejected_policy - torch.log(
                    alpha * sumlogprobs_rejected_policy.exp() +  (1 - alpha) * sumlogprobs_rejected_ref.exp() + 1e-36)
                log_importance_weights = ((importance_chosen + importance_rejected) * args.log_density_ratio_mult)
                importance_weights = log_importance_weights.exp()
                iw_normalizer = importance_weights.mean()
                iw_normalizer = iw_normalizer.to(accelerator.device)
            else:
                # log_importance_weights = ((dataset["log_density_ratio_chosen"] + dataset["log_density_ratio_rejected"]) * args.log_density_ratio_mult)
                log_importance_weights = (sumlogprobs_chosen_policy - sumlogprobs_chosen_ref + sumlogprobs_rejected_policy - sumlogprobs_rejected_ref) * args.log_density_ratio_mult
                importance_weights = log_importance_weights.exp()
                iw_normalizer = importance_weights.mean()
                iw_normalizer = iw_normalizer.to(accelerator.device)
            
            if accelerator.is_main_process:
                plt.hist(log_importance_weights, label='scaled_log_weights', fill=False, histtype='step', bins=20)
                plt.hist(raw_log_importance_weights, label='raw_log_weights', fill=False, histtype='step', bins=20)
                plt.legend()
                plt.xlabel('log density ratio')
                plt.ylabel('count')
                plot_fp_comb = os.path.join(args.reward_model_path, 'density_ratio_hist.png')
                plt.savefig(plot_fp_comb)
                plt.close()
                print(f'plotted to {plot_fp_comb}')
                plt.hist(log_importance_weights, label='scaled_log_weights', fill=False, histtype='step', bins=20)
                plt.legend()
                plt.xlabel('log density ratio')
                plt.ylabel('count')
                plot_fp = os.path.join(args.reward_model_path, 'density_ratio_hist_processed.png')
                plt.savefig(plot_fp)
                plt.close()
                print(f'plotted to {plot_fp}')
        else:
            dataset = dataset.with_format(
                "torch",
                columns=[
                    "query_token",
                    "chosen_token",
                    "query_chosen_token",
                    "rejected_token",
                    "query_rejected_token",
                ],
            )
    else:
        dataset = load_dataset(args.label_dataset, split="train")
        dataset = dataset.shuffle(seed=local_seed)
        dataset = dataset.select(range(args.total_episodes))
        dataset = dataset.with_format(
            "torch",
            columns=[
                "query_token",
                "chosen_token",
                "query_chosen_token",
                "rejected_token",
                "query_rejected_token",
                "batch",
                "split",
            ],
        )
    do_iw_val = args.iw_val_ds_path is not None
    if do_iw_val:
        if use_numpy_ds:
            val_dataset_dict = np.load(args.iw_val_ds_path, allow_pickle=True).item()
            val_dataset = Dataset.from_dict(val_dataset_dict)
        else:
            val_dataset = load_from_disk(args.iw_val_ds_path)
        val_dataset = val_dataset.shuffle(seed=args.seed)
        val_dataset = val_dataset.select(range(4096))
        val_dataset = val_dataset.with_format(
            "torch",
            columns=[
                "query_token",
                "chosen_token",
                "query_chosen_token",
                "rejected_token",
                "query_rejected_token",
            ],
        )
        iw_val_dataloader = DataLoader(val_dataset, batch_size=args.local_eval_batch_size)

    dataloader = DataLoader(dataset, batch_size=args.local_micro_batch_size)
    eval_datasets = []
    eval_dataloaders = {}
    for split in ["validation", "validation_cnndm"]:
        validation_dataset = load_dataset(args.label_dataset, split=split).flatten()
        validation_dataset = validation_dataset.with_format(
            "torch",
            columns=[
                "query_token",
                "choice",
                "chosen_token",
                "query_chosen_token",
                "rejected_token",
                "query_rejected_token",
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
        accelerator.print("The number of samples in validation_dataset", len(validation_dataset))
    accelerator.print("The number of samples in dataset", len(dataset))
    args.total_episodes = len(dataset)
    args.num_updates = args.total_episodes // args.batch_size

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    console = Console(force_terminal=True)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    if accelerator.is_main_process:
        if args.track:
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
            if importance_weighted:
                    wandb.log({"processed_density_ratio_hist_combined": wandb.Image(plot_fp)})
                    wandb.log({"density_ratio_hist_combined": wandb.Image(plot_fp_comb)})
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
    scalar_model_config = ScalarModelConfig(
        base_model=args.sft_model_path,
        base_config=model_config,
        hidden_size=model_config.hidden_size,
    )
    if len(args.reward_model_path) == 0:
        model: PreTrainedModel = ScalarModel(scalar_model_config)
    else:
        model: PreTrainedModel = ScalarModel.from_pretrained(
            args.reward_model_path,
            trust_remote_code=True,
        )
    disable_dropout(model)
    if accelerator.is_main_process:
        pprint(model_config)
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
    if do_iw_val:
        iw_val_dataloader = accelerator.prepare(iw_val_dataloader)
    torch.manual_seed(local_seed)  # reset the local seed again

    accelerator.print("===training model===")
    losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
    accuracies = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_preferreds = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_rejecteds = torch.zeros((args.gradient_accumulation_steps,), device=device)
    model.train()
    gradient_accumulation_idx = 0
    global_step = 0
    update = 0
    for epoch in tqdm(range(args.num_train_epochs), desc='epoch', dynamic_ncols=True):
        accelerator.print(f"epoch: {epoch}")
        for data in tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch}/{args.num_train_epochs} Step:', dynamic_ncols=True):
            update += 1
            global_step += args.micro_batch_size
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            with accelerator.accumulate(model):
                predicted_reward = get_reward(model, query_responses, tokenizer)
                chosen_rewards = predicted_reward[:data['query_chosen_token'].shape[0]]
                rejected_rewards = predicted_reward[data['query_chosen_token'].shape[0]:]
                accuracy = (chosen_rewards > rejected_rewards).float().mean()
                if importance_weighted:
                    unweighted_loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
                    if args.relative_density_factor > 0.0:
                        alpha = args.relative_density_factor
                        importance_chosen = data["sumlogprobs_chosen_policy"] - torch.log(alpha * data["sumlogprobs_chosen_policy"].exp() + 
                                                                                            (1 - alpha) * data["sumlogprobs_chosen_ref"].exp() + 1e-36)
                        importance_rejected = data["sumlogprobs_rejected_policy"] - torch.log(alpha * data["sumlogprobs_rejected_policy"].exp() + 
                                                                                                (1 - alpha) * data["sumlogprobs_rejected_ref"].exp() + 1e-36)
                        importance_weight = torch.exp(args.log_density_ratio_mult * (importance_chosen + importance_rejected))
                    else:
                        importance_weight = torch.exp(args.log_density_ratio_mult * (data["log_density_ratio_chosen"] + data["log_density_ratio_rejected"]))
                    if args.batch_normalize_importance_weights:
                        all_importance_weights = accelerator.gather(importance_weight)
                        iw_sum = all_importance_weights.sum()
                        importance_weight = importance_weight / iw_sum 
                    if args.dataset_normalize_importance_weights:
                        importance_weight = importance_weight / iw_normalizer
                    loss = (unweighted_loss * importance_weight).mean()
                else:
                    unweighted_loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
                    loss = unweighted_loss.mean()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            losses[gradient_accumulation_idx] = loss
            accuracies[gradient_accumulation_idx] = accuracy
            reward_preferreds[gradient_accumulation_idx] = chosen_rewards.mean()
            reward_rejecteds[gradient_accumulation_idx] = rejected_rewards.mean()
            gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
            if update > 1 and (update - 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()
                train_accuracy = accelerator.gather(accuracies).mean().item()
                writer.add_scalar("train/rm/loss", accelerator.gather(losses).mean().item(), global_step)
                writer.add_scalar("train/rm/unweighted_loss", accelerator.gather(unweighted_loss.mean()).mean().item(), global_step)
                writer.add_scalar("train/rm/accuracy", train_accuracy, global_step)
                writer.add_scalar(
                    "train/rm/chosen_rewards", accelerator.gather(reward_preferreds).mean().item(), global_step
                )
                writer.add_scalar("train/rm/rejected_rewards", accelerator.gather(reward_rejecteds).mean().item(), global_step)
                writer.add_scalar("train/rm/lr", scheduler.get_last_lr()[0], global_step)
                accelerator.print(
                    f"{train_accuracy=}, {scheduler.get_last_lr()=}, {optimizer.param_groups[0]['lr']=}, {update=}"
                )
            if do_iw_val and (update - 1) % 50 == 0:
                iw_data_accuracy, iw_chosen_reward, iw_rejected_reward, iw_loss = evaluate_iwds_accuracy(args, accelerator, tokenizer, model, iw_val_dataloader)
                accelerator.print(f"{iw_data_accuracy=}, {iw_chosen_reward=}, {iw_rejected_reward=}, {iw_loss=}")
                writer.add_scalar("val/rm/iw_data_accuracy", iw_data_accuracy, global_step)
                writer.add_scalar("val/rm/iw_chosen_reward", iw_chosen_reward, global_step)
                writer.add_scalar("val/rm/iw_rejected_reward", iw_rejected_reward, global_step)
                writer.add_scalar("val/rm/iw_loss", iw_loss, global_step)

    del loss, unweighted_loss, accuracy, chosen_rewards, rejected_rewards, query_responses, predicted_reward
    del optimizer, scheduler, losses, accuracies, reward_preferreds, reward_rejecteds
    torch.cuda.empty_cache()
    if args.run_eval:
        for eval_split in eval_dataloaders:
            evaluate_df = evaluate(args, accelerator, tokenizer, model, eval_dataloaders[eval_split])
            for split, row in evaluate_df[["split", "accuracy"]].groupby(["split"]).mean().iterrows():
                writer.add_scalar(f"eval/rm/{eval_split}/accuracy/split/{split}", row["accuracy"], global_step)
                accelerator.print(f"eval/rm/{eval_split}/accuracy/split/{split}: {row['accuracy']}")
            for batch, row in evaluate_df[["batch", "accuracy"]].groupby(["batch"]).mean().iterrows():
                writer.add_scalar(f"eval/rm/{eval_split}/accuracy/batch/{batch}", row["accuracy"], global_step)
                accelerator.print(f"eval/rm/{eval_split}/accuracy/batch/{batch}: {row['accuracy']}")
            for confi, row in evaluate_df[["confidence", "accuracy"]].groupby(["confidence"]).mean().iterrows():
                writer.add_scalar(f"eval/rm/{eval_split}/accuracy/confidence/{confi}", row["accuracy"], global_step)
                accelerator.print(f"eval/rm/{eval_split}/accuracy/confidence/{confi}: {row['accuracy']}")
            writer.add_scalar(f"eval/rm/{eval_split}/accuracy", evaluate_df["accuracy"].mean(), global_step)
            accelerator.print(f"eval/rm/{eval_split}/accuracy: {evaluate_df['accuracy'].mean()}")
            if accelerator.is_main_process:
                os.makedirs(f"eval_tables/{args.run_name}", exist_ok=True)
                evaluate_df.to_csv(f"eval_tables/{args.run_name}/eval_{eval_split}_{update}.csv")
                if args.track:
                    wandb.log({f"samples/{eval_split}/query_responses": wandb.Table(dataframe=evaluate_df)}, step=update)
            del evaluate_df
            torch.cuda.empty_cache()

    norm_dataset = load_dataset(args.query_dataset, split="train")
    norm_dataset = norm_dataset.with_format("torch", columns=["query_token", "reference_response_token", "query_reference_response_token"])
    norm_dataset = norm_dataset.shuffle(seed=local_seed)
    norm_dataloader = DataLoader(norm_dataset, batch_size=args.local_eval_batch_size)
    items = defaultdict(list)
    norm_dataloader = accelerator.prepare(norm_dataloader)
    rtol = 1e-2
    if args.output_dir:
        with torch.no_grad():
            for data in tqdm(norm_dataloader, desc='Normalization Dataset', dynamic_ncols=True):
                reference_responses = data["reference_response_token"].to(device, non_blocking=True)
                queries = data["query_token"].to(device, non_blocking=True)
                query_responses = data["query_reference_response_token"].to(device, non_blocking=True)
                cat_query_responses = torch.cat((queries, reference_responses), dim=1)
                cat_predicted_reward = get_reward(model, cat_query_responses, tokenizer, context_length=queries.shape[1])
                predicted_reward = get_reward(model, query_responses, tokenizer)
                unexpecte_reward_diff = predicted_reward - cat_predicted_reward
                unexpecte_reward_diff_gt_rtol = unexpecte_reward_diff.abs() > rtol
                unexpecte_reward_diff = accelerator.gather(unexpecte_reward_diff)
                unexpecte_reward_diff_gt_rtol = accelerator.gather(unexpecte_reward_diff_gt_rtol)
                predicted_reward = accelerator.gather(predicted_reward)
                queries = accelerator.gather(queries)
                reference_responses = accelerator.gather(reference_responses)
                for i in range(len(predicted_reward)):
                    items["query"].append(tokenizer.decode(queries[i], skip_special_tokens=True))
                    items["reference_response"].append(tokenizer.decode(reference_responses[i]))
                    items["predicted_reward"].append(predicted_reward[i].item())
                    items["unexpecte_reward_diff"].append(unexpecte_reward_diff[i].item())
                    items["unexpecte_reward_diff_gt_rtol"].append(unexpecte_reward_diff_gt_rtol[i].item())

    if accelerator.is_main_process and args.output_dir:
        norm_df = pd.DataFrame(items)
        os.makedirs(f"eval_tables/{args.run_name}", exist_ok=True)
        norm_ds = Dataset.from_pandas(norm_df)
        # norm_df.to_csv(f"eval_tables/{args.run_name}/eval_{update}_normalized.csv")
        norm_ds.save_to_disk(f"eval_tables/{args.run_name}/eval_{update}_normalized")
        if args.track:
            wandb.log({"samples/normalized": wandb.Table(dataframe=norm_df)}, step=update)
        stats = {
            "mean": norm_df["predicted_reward"].mean(),
            "std": norm_df["predicted_reward"].std(),
            "max": norm_df["predicted_reward"].max(),
            "min": norm_df["predicted_reward"].min(),
            "unexpecte_reward_diff_mean": norm_df["unexpecte_reward_diff"].mean(),
            "unexpecte_reward_diff_gt_rtol_mean": norm_df["unexpecte_reward_diff_gt_rtol"].mean(),
        }
        for stat_name, stat_value in stats.items():
            writer.add_scalar(f"eval/rm/normalized_{stat_name}", stat_value, global_step)
            accelerator.print(f"Normalized Reward {stat_name.capitalize()}: {stat_value}")

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
            unwrapped.config.bias = norm_df["predicted_reward"].mean()
            unwrapped.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=True,
            )
