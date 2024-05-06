MODEL_PATH=/cpfs01/shared/GAIR/GAIR_hdd/yguo/ckpts/Qwen/Qwen1.5-7B-Chat
TOKENIZER_PATH=$MODEL_PATH
export CUDA_VISIBLE_DEVICES=4,5,6,7


# python run.py --models vllm_qwen_1_5_7b_tulu2 --datasets ceval_gen math_gen
# python run.py --models vllm_llama2_7b_tulu2  --datasets mmlu_gen

# python run.py --models vllm_qwen_1_5_7b_metamath_full  --datasets math_gen gsm8k_gen

# python run.py --models vllm_qwen_1_5_7b_tulu2_math_code_enhanced --datasets bbh_gen humaneval_gen mbpp_gen mmlu_gen 


python run.py --models vllm_qwen_1_5_7b_tulu2_math_code_enhanced --datasets tydiqa_gen