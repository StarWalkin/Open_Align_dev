from vllm_utils import *
from prompt_utils import *
from utils import *

import argparse

ckpt_dir = "/cpfs01/shared/GAIR/GAIR_hdd/ckpts"
mapping_chat = {
    "llama-2-7b-chat": {"path": f"{ckpt_dir}/llama-2/7b-chat", "wrapper_type": "llama-2-chat", "AB_token_ids":[319, 350]},
    "llama-2-13b-chat": {"path": f"{ckpt_dir}/llama-2/llama-2-13b-chat-hf", "wrapper_type": "llama-2-chat","AB_token_ids":[319, 350]},
    "llama-2-70b-chat": {"path": f"{ckpt_dir}/llama-2/llama-2-70b-chat-hf", "wrapper_type": "llama-2-chat","AB_token_ids":[319, 350]},

    "vicuna-7b-v1.5": {"path": f"{ckpt_dir}/vicuna/7b-v1.5", "wrapper_type": "vicuna","AB_token_ids":[319, 350]},
    "vicuna-13b-v1.5": {"path": f"{ckpt_dir}/vicuna/13b-v1.5", "wrapper_type": "vicuna","AB_token_ids":[319, 350]},
    "wizardLM-13b-v1.2": {"path": f"{ckpt_dir}/WizardLM/WizardLM-13B-V1.2", "wrapper_type": "vicuna","AB_token_ids":[319, 350]},
    "wizardLM-70b-v1.0": {"path": f"{ckpt_dir}/WizardLM/WizardLM-70B-V1.0", "wrapper_type": "vicuna","AB_token_ids":[319, 350]},

    "tulu-2-dpo-7b": {"path": f"{ckpt_dir}/tulu/2-dpo-7b", "wrapper_type": "tulu","AB_token_ids":[319, 350]},
    "tulu-2-dpo-13b": {"path": f"{ckpt_dir}/tulu/2-dpo-13b", "wrapper_type": "tulu","AB_token_ids":[319, 350]},
    "tulu-2-dpo-70b": {"path": f"{ckpt_dir}/tulu/2-dpo-70b", "wrapper_type": "tulu","AB_token_ids":[319, 350]},

    "mistral-7b-instruct-v0.1": {"path": f"{ckpt_dir}/mistralai/Mistral-7B-Instruct-v0.1", "wrapper_type": "mistral","AB_token_ids":[330, 365]},
    "mistral-7b-instruct-v0.2": {"path": f"{ckpt_dir}/mistralai/Mistral-7B-Instruct-v0.2", "wrapper_type": "mistral","AB_token_ids":[330, 365]},
    "mistral-8x7b-instruct-v0.1": {"path": f"{ckpt_dir}/mistralai/mixtral-8x7b-instruct-v0.1", "wrapper_type": "mistral","AB_token_ids":[330, 365]},

    "yi-6b-chat": {"path": f"{ckpt_dir}/yi/Yi-6B-Chat", "wrapper_type": "yi","AB_token_ids":[647, 690]},
    "yi-34b-chat": {"path": f"{ckpt_dir}/yi/Yi-34B-Chat", "wrapper_type": "yi","AB_token_ids":[647, 690]},

    "qwen-7b-chat": {"path": f"{ckpt_dir}/Qwen/Qwen-7B-Chat", "wrapper_type": "qwen","AB_token_ids":[362,425]},
    "qwen-14b-chat": {"path": f"{ckpt_dir}/Qwen/Qwen-14B-Chat", "wrapper_type": "qwen","AB_token_ids":[362,425]},
    "qwen-72b-chat": {"path": f"{ckpt_dir}/Qwen/Qwen-72B-Chat", "wrapper_type": "qwen","AB_token_ids":[362,425]},

    "zephyr-7b-beta": {"path": f"{ckpt_dir}/zephyr/zephyr-7b-beta", "wrapper_type": "zephyr","AB_token_ids":[330, 365]},
    "zephyr-7b-alpha": {"path": f"{ckpt_dir}/zephyr/zephyr-7b-alpha", "wrapper_type": "zephyr","AB_token_ids":[330, 365]},

    "gemma-7b-it": {"path": f"{ckpt_dir}/google/gemma-7b-it", "wrapper_type": "gemma","AB_token_ids":[235280, 235305]},
    "gemma-2b-it": {"path": f"{ckpt_dir}/google/gemma-2 b-it", "wrapper_type": "gemma","AB_token_ids":[235280, 235305]},
}

mapping_base = {
    "llama-2-7b": {"path": f"{ckpt_dir}/llama-2/7b", "wrapper_type": "llama-2", "AB_token_ids":[319, 350]},
    "llama-2-13b": {"path": f"{ckpt_dir}/llama-2/13b", "wrapper_type": "llama-2", "AB_token_ids":[319, 350]},
    "llama-2-70b": {"path": f"{ckpt_dir}/llama-2/70b", "wrapper_type": "llama-2", "AB_token_ids":[319, 350]},
    "mistral-7b": {"path": f"{ckpt_dir}/mistralai/Mistral-7B-v0.1", "wrapper_type": "mistral","AB_token_ids":[330, 365]},
    "mistral-8x7b": {"path": f"{ckpt_dir}/mistralai/Mixtral-8x7B-v0.1", "wrapper_type": "mistral","AB_token_ids":[330, 365]},
    "yi-6b": {"path": f"{ckpt_dir}/yi/Yi-6B-Llama", "wrapper_type": "yi","AB_token_ids":[647, 690]},
    "yi-34b": {"path": f"{ckpt_dir}/yi/Yi-34B-Llama", "wrapper_type": "yi","AB_token_ids":[647, 690]},
    "qwen-7b": {"path": f"{ckpt_dir}/Qwen/Qwen-7B", "wrapper_type": "qwen", "AB_token_ids": [362,425]},
    "qwen-14b": {"path": f"{ckpt_dir}/Qwen/Qwen-14B", "wrapper_type": "qwen", "AB_token_ids": [362,425]},
    "qwen-72b": {"path": f"{ckpt_dir}/Qwen/Qwen-72B", "wrapper_type": "qwen", "AB_token_ids": [362,425]},
}

mapping_extra = {
    "llama-2-7b-chat-dpo-positive-chatgpt":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/llama2_7b_preference_gpt_3.5","wrapper_type": "llama-2-chat", "AB_token_ids":[319, 350]},
    "llama-2-7b-chat-dpo-negative-chatgpt": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/llama2_7b_preference_gpt_3.5_inv", "wrapper_type": "llama-2-chat", "AB_token_ids": [319, 350]},
    "llama-2-7b-chat-dpo-positive-human":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/llama2_7b_preference_human","wrapper_type": "llama-2-chat", "AB_token_ids":[319, 350]},
    "llama-2-7b-chat-dpo-negative-human": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/llama2_7b_preference_human_inv", "wrapper_type": "llama-2-chat", "AB_token_ids": [319, 350]},
    "llama-2-7b-chat-dpo-positive-gpt4":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/llama2_7b_preference_gpt_4","wrapper_type": "llama-2-chat", "AB_token_ids":[319, 350]},
    "llama-2-7b-chat-dpo-negative-gpt4": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/llama2_7b_preference_gpt_4_inv", "wrapper_type": "llama-2-chat", "AB_token_ids": [319, 350]},

    "llama-2-13b-chat-dpo-positive-chatgpt":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/llama2_13b_preference_gpt_3.5","wrapper_type": "llama-2-chat", "AB_token_ids":[319, 350]},
    "llama-2-13b-chat-dpo-negative-chatgpt": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/llama2_13b_preference_gpt_3.5_inv", "wrapper_type": "llama-2-chat", "AB_token_ids": [319, 350]},
    "llama-2-13b-chat-dpo-positive-human":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/llama2_13b_preference_human","wrapper_type": "llama-2-chat", "AB_token_ids":[319, 350]},
    "llama-2-13b-chat-dpo-negative-human": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/llama2_13b_preference_human_inv", "wrapper_type": "llama-2-chat", "AB_token_ids": [319, 350]},
    "llama-2-13b-chat-dpo-positive-gpt4":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/llama2_13b_preference_gpt_4","wrapper_type": "llama-2-chat", "AB_token_ids":[319, 350]},
    "llama-2-13b-chat-dpo-negative-gpt4": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/llama2_13b_preference_gpt_4_inv", "wrapper_type": "llama-2-chat", "AB_token_ids": [319, 350]},
}

# put mapping_chat and mapping_base together to get mapping
mapping = {**mapping_chat, **mapping_base, **mapping_extra}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",default="llama-2-7b-chat")
    parser.add_argument("--preference_way",default="binary_choice")
    parser.add_argument("--ask_AB_prompt",default="naive")
    parser.add_argument("--change_AB",default=False, action="store_true")
    parser.add_argument("--end_pos",type=int,default=1000000000,)
    args = parser.parse_args()

    # get model name from input argument
    model_name = args.model_name
    assert model_name in mapping
    # information of the model
    model_dir = mapping[model_name]["path"]
    wrapper_type = mapping[model_name]["wrapper_type"]
    r1r2_token_ids = mapping[model_name]["AB_token_ids"]
    # we use default: binary choice here
    way = args.preference_way


    # initialize the vllm engine with given model
    vllm_engine = VllmEngine(model_name_or_dir=model_dir)
    # vllm_engine = None
    tokenizer = vllm_engine.llm.get_tokenizer()
    # tokenizer = None

    # get the pairwise data
    data = read_all(
        "/cpfs01/user/liupengfei/jlli/pjs/alignment-decomposition/shared/data/chatbot_arena_shuffled_no-tie_group_balanced.jsonl")

    if way == "binary_choice":
        all_wrapped_q_rpairs = []
        AB_flag = "AB" if not args.change_AB else "BA"
    # start from what has been done
        start_pos = len(read_all(
             f"/cpfs01/user/liupengfei/yguo/Open_Align_dev/preference_dissection/{model_name}/direct_ask_{args.ask_AB_prompt}_preferences_{AB_flag}.jsonl"))

        print(data[500:520])
        for item in data[100:105]:
            query = item["prompt"]
            r1 = item["response_a"]
            r2 = item["response_b"]
            if args.change_AB:
                r1, r2 = r2, r1
            sysmsg, wrapped_p_rpair = wrapper_p_rpair(wrapper_type=args.ask_AB_prompt, query=query, response_1=r1, response_2=r2,)
            all_wrapped_q_rpairs.append(wrapped_p_rpair)
            # print(wrapped_p_rpair)
            # raise ValueError


        print("1111111111111111111111111111111111111111111111111111111111")
        pprint(all_wrapped_q_rpairs[100])
        pprint(all_wrapped_q_rpairs[101])
        # pprint(all_wrapped_q_rpairs[102])
        print("1111111111111111111111111111111111111111111111111111111111")

        AB_probs = vllm_engine.get_all_first_generated_token_probs(all_wrapped_q_rpairs, r1r2_token_ids=r1r2_token_ids)
        write_jsonl(AB_probs, f"/cpfs01/user/liupengfei/yguo/Open_Align_dev/preference_dissection/{model_name}/direct_ask_{args.ask_AB_prompt}_preferences_{AB_flag}.jsonl",mode="a")
