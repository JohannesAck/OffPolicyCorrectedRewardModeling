#!/bin/bash
set -e
SEED=1
if [ -z "$MODEL" ]; then
    # MODEL=EleutherAI/pythia-6.9b-deduped
    # MODEL=EleutherAI/pythia-2.8b-deduped
    # MODEL=EleutherAI/pythia-1b-deduped
    # MODEL=EleutherAI/pythia-70M-deduped
    MODEL=EleutherAI/pythia-14M
    # MODEL=EleutherAI/pythia-160m-deduped
fi
LR=3e-6
SFT_TEMP=1.0
KL_COEF=0.05
PPO_STEPS=1000

REWARD_MODEL_PATH=models/$MODEL/reward_model_$SEED
REWARD_MODEL2_PATH=models/$MODEL/reward_model2_${SEED}
SFT_MODEL_PATH=models/$MODEL/sft_model_$SEED
POLICY_MODEL_PATH=models/$MODEL/policy_model_${SEED}
POLICY_MODEL2_PATH=models/$MODEL/policy_model2_${SEED}

# these environment variable may not be necessary for you
export OMP_NUM_THREADS=16
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1 

# vary the following parameters to fit your GPU memory

gpu_count=$(nvidia-smi --list-gpus | wc -l)
echo "Using $gpu_count GPUs"

function DoSource() { source /root/.cache/pypoetry/virtualenvs/ocrm-*-py3.10/bin/activate ; }
DoSource
echo sourced.

main_process_port=23524
# Check if the port is in use
while netstat -tuln | grep -q ":$main_process_port "; do
    # Increment port number
    ((main_process_port++))
done
echo "Found an unused port: $main_process_port"


SLEEPBTW_STEPS=10

iwreward_local_batchsize=48
reward_local_batchsize=48
local_rollout_forward_batch_size=16
local_sft_sampling_batch_size=32
local_micro_batch_size=16
local_eval_batch_size=16
sft_batch_size=32
ppo_gradient_accumulation_steps=4
sft_gradient_accumulation_steps=1
reward_model_accumulation=1
# the batch sizes are not consistent for the debugging script. Change for actual training
if [ $gpu_count == 4 ]; then
    ppo_gradient_accumulation_steps=8
    sft_gradient_accumulation_steps=1
    reward_model_accumulation=1
fi
if [ $gpu_count == 2 ]; then
    ppo_gradient_accumulation_steps=16
    sft_gradient_accumulation_steps=2
    reward_model_accumulation=2
fi
if [ $gpu_count == 1 ]; then
    ppo_gradient_accumulation_steps=32
    sft_gradient_accumulation_steps=4
    reward_model_accumulation=4
fi


/root/.local/bin/poetry run accelerate launch --config_file deepspeed.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/sft.py \
    --gradient_accumulation_steps=$sft_gradient_accumulation_steps \
    --local_micro_batch_size=$sft_batch_size \
    --base_model=$MODEL \
    --lr=$LR \
    --deepspeed \
    --track \
    --output_dir=$SFT_MODEL_PATH \
    --seed=$SEED

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeedNoOpt.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/sample_sftmodel.py \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --local_batch_size=$local_sft_sampling_batch_size \
    --seed=$SEED \
    --temperature=$SFT_TEMP \
    --num_samples=500

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeedNoOpt.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/gold_label_dataset.py \
    --dataset_fp=$SFT_MODEL_PATH/dataset_full \
    --local_batch_size=4
 
sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeed.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/reward.py \
    --gradient_accumulation_steps=$reward_model_accumulation \
    --local_micro_batch_size=$reward_local_batchsize \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --lr=$LR \
    --deepspeed \
    --track \
    --output_dir=$REWARD_MODEL_PATH \
    --local_eval_batch_size=$local_eval_batch_size \
    --no-run_eval \
    --custom_df_path=$SFT_MODEL_PATH/dataset_full_goldadded \
    --seed=$SEED

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeed.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/ppo.py \
    --exp_name=ppo1 \
    --local_rollout_forward_batch_size=$local_rollout_forward_batch_size \
    --gradient_accumulation_steps=$ppo_gradient_accumulation_steps \
    --local_micro_batch_size=$local_micro_batch_size \
    --base_model=$MODEL \
    --kl_penalty_model_path=$SFT_MODEL_PATH \
    --reward_model_path=$REWARD_MODEL_PATH \
    --lr=$LR \
    --deepspeed \
    --total_episodes=${PPO_STEPS} \
    --track \
    --output_dir=$POLICY_MODEL_PATH \
    --seed=$SEED

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --main_process_port=23523 --mixed_precision=bf16 --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/get_density_ratios.py \
    --dataset_fp=$SFT_MODEL_PATH/dataset_full_goldadded \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --ppo_model_path=$POLICY_MODEL_PATH \
    --temperature=$SFT_TEMP \
    --local_batch_size=64 \
    --output_name=dataset_full_densityratio

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeed.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/reward.py \
    --exp_name=reward2 \
    --gradient_accumulation_steps=$reward_model_accumulation \
    --local_micro_batch_size=$iwreward_local_batchsize \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --lr=$LR \
    --deepspeed \
    --track \
    --output_dir=$REWARD_MODEL2_PATH \
    --local_eval_batch_size=$local_eval_batch_size \
    --custom_df_path=$SFT_MODEL_PATH/dataset_full_densityratio \
    --log_density_ratio_mult=0.001 \
    --no-run_eval \
    --seed=$SEED

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeed.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/ppo.py \
    --exp_name=ppo2 \
    --local_rollout_forward_batch_size=$local_rollout_forward_batch_size \
    --gradient_accumulation_steps=$ppo_gradient_accumulation_steps \
    --local_micro_batch_size=$local_micro_batch_size \
    --base_model=$MODEL \
    --kl_penalty_model_path=$POLICY_MODEL_PATH \
    --kl_validation_model_path=$SFT_MODEL_PATH \
    --reward_model_path=$REWARD_MODEL2_PATH \
    --policy_init_model_path=$POLICY_MODEL_PATH \
    --lr=$LR \
    --deepspeed \
    --total_episodes=${PPO_STEPS} \
    --track \
    --output_dir=${POLICY_MODEL2_PATH} \
    --ppo.kl_coef=$KL_COEF \
    --seed=$SEED

