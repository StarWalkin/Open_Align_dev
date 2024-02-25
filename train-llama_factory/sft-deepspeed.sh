#!/usr/bin/env bash
# wandb key of yuan_alignment set here
# the config of wandb can be found in ds_config.json
export WANDB_API_KEY="a2ce506b19e91fc1224ec8c989d564ad8f5e40ba"

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&1
	exit 1
fi

# set some arguments here
ckpts_dir="/cpfs01/shared/GAIR/GAIR_hdd/ckpts"
output_dir="/cpfs01/user/liupengfei/yguo/ckpts"
BASE_MODEL=${ckpts_dir}/llama-2/7b
OUTPUT_DIR=${output_dir}/llama2-sft-tulu2
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

echo -e "\
      base model path: ${BASE_MODEL}\n\
      GPU number: ${NUM_GPUS}\n\
      batch size per GPU: ${BATCH_SIZE_PER_GPU}\n\
      gradient accumulation steps: ${GRADIENT_ACC_STEPS}\n\
      output path: ${OUTPUT_DIR}\n
      "

#mkdir -p "${OUTPUT_DIR}"
#OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
#if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
#	echo '*' >"${OUTPUT_DIR}/.gitignore"
#fi

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)


# the script is based on the setting in tulu2 paper, you can modify the setting here according to your own needs'
# for detailed guidance of the parameters, run python src/train_bash.py --help
# REMEMBER TO CHANGE adjust batch size when using less than 8 gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --do_train True\
    --model_name_or_path ${BASE_MODEL} \
    --dataset tulu-v2-sft-mixture \
    --template default \
    --cutoff_len 8192 \
    --finetuning_type full \
    --temperature 0 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --weight_decay 0.0 \
    --gradient_accumulation_steps ${GRADIENT_ACC_STEPS} \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --save_steps 1000 \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --plot_loss \
    --report_to "wandb"\
    --bf16 True \
    --tf32 True \
    --overwrite_output_dir

OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi