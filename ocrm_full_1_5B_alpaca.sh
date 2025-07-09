#!/bin/bash
set -e
SEED=$1
MODEL=Qwen/Qwen2.5-1.5B

# ALPACA_QUERY_DATASET=johannesack/alpaca_sft1738133725
ALPACA_QUERY_DATASET=johannesack/alpaca_qwen_sft1738158738

LR=3e-6
SFT_TEMP=1.0
KL_COEF=0.1
REWARD_MODEL_PATH=models/${MODEL}_alpaca/reward_model_$SEED
REWARD_MODEL2_PATH=models/${MODEL}_alpaca/reward_model2_$SEED
REWARD_MODEL3_PATH=models/${MODEL}_alpaca/reward_model3_${SEED}
SFT_MODEL_PATH=models/${MODEL}_alpaca/sft_model_$SEED
POLICY_MODEL_PATH=models/${MODEL}_alpaca/policy_model_$SEED
POLICY_MODEL2_PATH=models/${MODEL}_alpaca/policy_model2_$SEED
POLICY_MODEL3_PATH=models/${MODEL}_alpaca/policy_model3_${SEED}

export OMP_NUM_THREADS=16
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1



function DoSource() { source /root/.cache/pypoetry/virtualenvs/ocrm-*-py3.10/bin/activate ; }
DoSource
echo sourced.

main_process_port=23523
# Check if the port is in use
while netstat -tuln | grep -q ":$main_process_port "; do
    # Increment port number
    ((main_process_port++))
done
echo "Found an unused port: $main_process_port"

gpu_count=$(nvidia-smi --list-gpus | wc -l)
echo gpu count $gpu_count


SLEEPBTW_STEPS=60


iwreward_local_batchsize=8
reward_local_batchsize=8
local_rollout_forward_batch_size=8 # smaller fits better on GPU
local_sft_sampling_batch_size=8
local_micro_batch_size=2 # smaller fits better on GPU
local_eval_batch_size=4 # smaller fits better on GPU
sft_batch_size=8 
ppo_gradient_accumulation_steps=32 # bigger fits better on GPU
sft_gradient_accumulation_steps=2
reward_model_accumulation=2
DENSITY_BATCHSIZE=32
if [ $gpu_count == 4 ]; then
    ppo_gradient_accumulation_steps=64 # bigger fits better on GPU
    sft_gradient_accumulation_steps=4
    reward_model_accumulation=4
fi

/root/.local/bin/poetry run accelerate launch --config_file deepspeed.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/sft.py \
    --exp_name=sft_alpaca \
    --gradient_accumulation_steps=$sft_gradient_accumulation_steps \
    --local_micro_batch_size=$sft_batch_size \
    --base_model=$MODEL \
    --lr=$LR \
    --deepspeed \
    --track \
    --output_dir=$SFT_MODEL_PATH \
    --run_eval \
    --seed=$SEED \
    --query_dataset=$ALPACA_QUERY_DATASET \
    --response_length=106

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeedNoOpt.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/sample_sftmodel.py \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --local_batch_size=$local_sft_sampling_batch_size \
    --seed=$SEED \
    --temperature=$SFT_TEMP \
    --query_dataset=$ALPACA_QUERY_DATASET \
    --sampling_epochs=10 \
    --response_length=106

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeedNoOpt.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/gold_label_dataset.py \
    --dataset_fp=$SFT_MODEL_PATH/dataset_full \
    --local_batch_size=64 
 
sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeed.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/reward.py \
    --exp_name=reward_alpaca \
    --gradient_accumulation_steps=$reward_model_accumulation \
    --local_micro_batch_size=$reward_local_batchsize \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --lr=$LR \
    --deepspeed \
    --track \
    --output_dir=$REWARD_MODEL_PATH \
    --local_eval_batch_size=$local_eval_batch_size \
    --custom_df_path=$SFT_MODEL_PATH/dataset_full_goldadded \
    --query_dataset=$ALPACA_QUERY_DATASET \
    --seed=$SEED

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeed.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/ppo.py \
    --exp_name=ppo_alpaca \
    --local_rollout_forward_batch_size=$local_rollout_forward_batch_size \
    --gradient_accumulation_steps=$ppo_gradient_accumulation_steps \
    --local_micro_batch_size=$local_micro_batch_size \
    --base_model=$MODEL \
    --kl_penalty_model_path=$SFT_MODEL_PATH \
    --reward_model_path=$REWARD_MODEL_PATH \
    --lr=$LR \
    --deepspeed \
    --run_eval \
    --track \
    --print_sample_output_freq=5 \
    --output_dir=$POLICY_MODEL_PATH \
    --seed=$SEED \
    --query_dataset=$ALPACA_QUERY_DATASET \
    --response_length=106 \
    --ppo.kl_coef=$KL_COEF \
    --print_sample_output_freq=5


/root/.local/bin/poetry run accelerate launch --main_process_port=23523 --mixed_precision=bf16 --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/get_density_ratios.py \
    --dataset_fp=$SFT_MODEL_PATH/dataset_full_goldadded \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --ppo_model_path=$POLICY_MODEL_PATH \
    --temperature=$SFT_TEMP \
    --local_batch_size=$DENSITY_BATCHSIZE

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeed.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/reward.py \
    --exp_name=reward2_alpaca \
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
    --query_dataset=$ALPACA_QUERY_DATASET \
    --seed=$SEED

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeed.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/ppo.py \
    --exp_name=ppo2_alpaca \
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
    --run_eval \
    --track \
    --output_dir=${POLICY_MODEL2_PATH} \
    --ppo.kl_coef=$KL_COEF \
    --seed=$SEED \
    --query_dataset=$ALPACA_QUERY_DATASET \
    --response_length=106

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --main_process_port=23523 --mixed_precision=bf16 --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/get_density_ratios.py \
    --dataset_fp=$SFT_MODEL_PATH/dataset_full_goldadded \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --ppo_model_path=${POLICY_MODEL2_PATH} \
    --temperature=$SFT_TEMP \
    --local_batch_size=$DENSITY_BATCHSIZE \
    --output_name=dataset_full_densityratio2_retry


sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeed.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/reward.py \
    --exp_name=reward3_alpaca \
    --gradient_accumulation_steps=$reward_model_accumulation \
    --local_micro_batch_size=$iwreward_local_batchsize \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --lr=$LR \
    --deepspeed \
    --track \
    --output_dir=$REWARD_MODEL3_PATH \
    --local_eval_batch_size=$local_eval_batch_size \
    --custom_df_path=$SFT_MODEL_PATH/dataset_full_densityratio2_retry \
    --log_density_ratio_mult=0.001 \
    --query_dataset=$ALPACA_QUERY_DATASET \
    --seed=$SEED

sleep $SLEEPBTW_STEPS

/root/.local/bin/poetry run accelerate launch --config_file deepspeed.yaml --num_processes $gpu_count --main_process_port=$main_process_port \
    ocrm/ppo.py \
    --exp_name=ppo3_alpaca \
    --local_rollout_forward_batch_size=$local_rollout_forward_batch_size \
    --gradient_accumulation_steps=$ppo_gradient_accumulation_steps \
    --local_micro_batch_size=$local_micro_batch_size \
    --base_model=$MODEL \
    --kl_penalty_model_path=$POLICY_MODEL2_PATH \
    --kl_validation_model_path=$SFT_MODEL_PATH \
    --reward_model_path=$REWARD_MODEL3_PATH \
    --policy_init_model_path=$POLICY_MODEL2_PATH \
    --lr=$LR \
    --deepspeed \
    --run_eval \
    --track \
    --output_dir=${POLICY_MODEL3_PATH} \
    --ppo.kl_coef=$KL_COEF \
    --seed=$SEED \
    --query_dataset=$ALPACA_QUERY_DATASET \
    --response_length=106

sleep $SLEEPBTW_STEPS
