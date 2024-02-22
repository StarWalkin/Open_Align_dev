#!/usr/bin/env bash
#wandb key of yuan_alignment
export WANDB_API_KEY="a2ce506b19e91fc1224ec8c989d564ad8f5e40ba"

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&1
	exit 1
fi

ckpts_dir="/cpfs01/shared/GAIR/GAIR_hdd/ckpts"
output_dir="/cpfs01/user/liupengfei/yguo/ckpts"
OUTPUT_DIR=${output_dir}/llama2-sft-tulu2

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)


# the script is based on the setting in tulu2 paper, you can modify the setting here according to your own needs'
# for detailed guidance of the parameters, run python src/train_bash.py --help
# REMEMBER TO CHANGE adjust batch size when using less than 8 gpus
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --do_train True\
    --model_name_or_path ${ckpts_dir}/llama-2/13b/ \
    --dataset tulu-v2-sft-mixture \
    --template default \
    --cutoff_len 8192 \
    --finetuning_type full \
    --temperature 0 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --per_device_train_batch_size 16 \
    --weight_decay 0.0 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type constant_with_warmup??? \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --plot_loss \
    --report_to
    --bf16 True