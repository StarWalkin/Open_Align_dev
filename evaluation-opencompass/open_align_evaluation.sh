# run evaluation pipeline with opencompass to measure the general ability of the model
# Refer to the paper for how we define the general ability

MODEL_PATH=/cpfs01/shared/GAIR/GAIR_hdd/ckpts/Qwen/Qwen-7B-Chat/
TOKENIZER_PATH=$MODEL_PATH

# mmlu 0-shot evaluation
python run.py
--datasets mmlu_gen gsm8k_gen bbh_gen tydiqa_gen  \
--hf-path $MODEL_PATH \  # HuggingFace 模型地址
--tokenizer-path $TOKENIZER_PATH \  # HuggingFace 模型地址
--model-kwargs device_map='auto' trust_remote_code=True \  # 构造 model 的参数
--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \  # 构造 tokenizer 的参数
--max-out-len 100 \  # 模型能接受的最大序列长度
--max-seq-len 2048 \  # 最长生成 token 数
--batch-size 8 \  # 批次大小
--no-batch-padding \  # 不打开 batch padding，通过 for loop 推理，避免精度损失
--num-gpus 1  # 所需 gpu 数

## gsm8k 8-shot cot evaluation
#python run.py
#--datasets siqa_gen winograd_ppl \
#--hf-path $MODEL_PATH \  # HuggingFace 模型地址
#--tokenizer-path $TOKENIZER_PATH \  # HuggingFace 模型地址
#--model-kwargs device_map='auto' trust_remote_code=True \  # 构造 model 的参数
#--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \  # 构造 tokenizer 的参数
#--max-out-len 100 \  # 模型能接受的最大序列长度
#--max-seq-len 2048 \  # 最长生成 token 数
#--batch-size 8 \  # 批次大小
#--no-batch-padding \  # 不打开 batch padding，通过 for loop 推理，避免精度损失
#--num-gpus 1  # 所需 gpu 数
#
## BBH 3-shot cot evaluation
#python run.py
#--datasets siqa_gen winograd_ppl \
#--hf-path $MODEL_PATH \  # HuggingFace 模型地址
#--tokenizer-path $TOKENIZER_PATH \  # HuggingFace 模型地址
#--model-kwargs device_map='auto' trust_remote_code=True \  # 构造 model 的参数
#--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \  # 构造 tokenizer 的参数
#--max-out-len 100 \  # 模型能接受的最大序列长度
#--max-seq-len 2048 \  # 最长生成 token 数
#--batch-size 8 \  # 批次大小
#--no-batch-padding \  # 不打开 batch padding，通过 for loop 推理，避免精度损失
#--num-gpus 1  # 所需 gpu 数
#
## TydiQA 1-shot evaluation
#--datasets siqa_gen winograd_ppl \
#--hf-path $MODEL_PATH \  # HuggingFace 模型地址
#--tokenizer-path $TOKENIZER_PATH \  # HuggingFace 模型地址
#--model-kwargs device_map='auto' trust_remote_code=True \  # 构造 model 的参数
#--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \  # 构造 tokenizer 的参数
#--max-out-len 100 \  # 模型能接受的最大序列长度
#--max-seq-len 2048 \  # 最长生成 token 数
#--batch-size 8 \  # 批次大小
#--no-batch-padding \  # 不打开 batch padding，通过 for loop 推理，避免精度损失
#--num-gpus 1  # 所需 gpu 数
#
## Codex-Eval
#python run.py
#--datasets siqa_gen winograd_ppl \
#--hf-path $MODEL_PATH \  # HuggingFace 模型地址
#--tokenizer-path $TOKENIZER_PATH \  # HuggingFace 模型地址
#--model-kwargs device_map='auto' trust_remote_code=True \  # 构造 model 的参数
#--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \  # 构造 tokenizer 的参数
#--max-out-len 100 \  # 模型能接受的最大序列长度
#--max-seq-len 2048 \  # 最长生成 token 数
#--batch-size 8 \  # 批次大小
#--no-batch-padding \  # 不打开 batch padding，通过 for loop 推理，避免精度损失
#--num-gpus 1  # 所需 gpu 数
#
## ToxiGen evaluation
#python run.py
#--datasets siqa_gen winograd_ppl \
#--hf-path $MODEL_PATH \  # HuggingFace 模型地址
#--tokenizer-path $TOKENIZER_PATH \  # HuggingFace 模型地址
#--model-kwargs device_map='auto' trust_remote_code=True \  # 构造 model 的参数
#--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \  # 构造 tokenizer 的参数
#--max-out-len 100 \  # 模型能接受的最大序列长度
#--max-seq-len 2048 \  # 最长生成 token 数
#--batch-size 8 \  # 批次大小
#--no-batch-padding \  # 不打开 batch padding，通过 for loop 推理，避免精度损失
#--num-gpus 1  # 所需 gpu 数
#
## Truthful QA evaluation
#python run.py
#--datasets siqa_gen winograd_ppl \
#--hf-path $MODEL_PATH \  # HuggingFace 模型地址
#--tokenizer-path $TOKENIZER_PATH \  # HuggingFace 模型地址
#--model-kwargs device_map='auto' trust_remote_code=True \  # 构造 model 的参数
#--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True \  # 构造 tokenizer 的参数
#--max-out-len 100 \  # 模型能接受的最大序列长度
#--max-seq-len 2048 \  # 最长生成 token 数
#--batch-size 8 \  # 批次大小
#--no-batch-padding \  # 不打开 batch padding，通过 for loop 推理，避免精度损失
#--num-gpus 1  # 所需 gpu 数
#
#
