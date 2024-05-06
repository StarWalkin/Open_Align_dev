# run evaluation pipeline with opencompass to measure the general ability of the model
# Refer to the paper for how we define the general ability

MODEL_PATH=/cpfs01/shared/GAIR/GAIR_hdd/yguo/ckpts/Qwen/Qwen1.5-7B-Chat
TOKENIZER_PATH=$MODEL_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
# python run.py --models vllm_qwen_1_5_7b_chat --datasets teval_en_gen
# python run.py --models vllm_llama2_7b_tulu2 --datasets bbh_gen tydiqa_gen hellaswag_gen mmlu_gen
# python run.py --models vllm_qwen_1_5_7b_tulu2 --datasets mmlu_gen bbh_gen
# python run.py --models vllm_qwen_1_5_7b_base --datasets gsm8k_gen
# python run.py --models vllm_llama2_7b_base --datasets gsm8k_gen
# python run.py --models vllm_qwen_1_5_7b_base --datasets tydiqa_gen
# python run.py --models hf_qwen1_5_7b_chat --datasets longbench
# python run.py --models vllm_qwen_1_5_7b_chat --datasets longbench

export OPENAI_API_KEY=sk-rg8izdOSsK2tLXLt124865B116214e5bAa98C0Bc0c0b6aBc
export OPENAI_API_BASE=https://lonlie.plus7.plus/v1


# python run.py --models vllm_qwen_1_5_7b_tulu2 --datasets TheoremQA_gen
# python run.py --models vllm_llama2_7b_tulu2 vllm_llama2_7b_chat --datasets TheoremQA_gen mbpp_gen

# CUDA_VISIVLE_DEVICES=0,1,2,3 python run.py --datasets teval_en_gen --num-gpus 1 --models vllm_qwen_1_5_7b_chat vllm_qwen_1_5_7b_tulu2

# python run.py --models vllm_qwen_1_5_7b_metamath_50k --datasets math_gen 
# python run.py --models vllm_qwen_1_5_7b_base --datasets math_gen 

# python run.py --models vllm_qwen_1_5_7b_metamath_50k_3epoch --datasets math_gen

python run.py --models vllm_qwen_1_5_7b_tulu2_math_code_enhanced --datasets gsm8k_gen math_gen triviaqa_gen


# python run.py --models vllm_qwen_1_5_7b_chat --datasets humaneval_gen --debug
# mmlu 0-shot evaluation
# python run.py --datasets mmlu_gen gsm8k_gen bbh_gen --hf-path $MODEL_PATH --tokenizer-path $TOKENIZER_PATH --model-kwargs device_map='auto' trust_remote_code=True --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False trust_remote_code=True --max-out-len 200 --max-seq-len 4096 --batch-size 8 --no-batch-padding --num-gpus 4
# python run.py --models vllm_llama2_7b_chat vllm_qwen_1_5_7b_chat vllm_llama2_7b_tulu2 --datasets mmlu_gen bbh_gen gsm8k_gen hellaswag_gen teval_en_gen tydiqa_gen
# python run.py --models vllm_llama2_7b_chat --datasets mmlu_gen bbh_gen
# python run.py --models vllm_qwen_1_5_7b_chat --datasets mmlu_gen bbh_gen gsm8k_gen hellaswag_gen teval_en_gen tydiqa_gen
# python run.py --models vllm_llama2_7b_tulu2 --datasets mmlu_gen bbh_gen gsm8k_gen hellaswag_gen teval_en_gen tydiqa_gen
# python run.py --models vllm_llama2_7b_chat --datasets mmlu_gen bbh_gen gsm8k_gen hellaswag_gen teval_en_gen tydiqa_gen
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
