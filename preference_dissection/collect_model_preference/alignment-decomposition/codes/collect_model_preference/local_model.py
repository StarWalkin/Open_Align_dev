from shared.codes.vllm_utils import *
from shared.codes.prompt_utils import *

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
mapping = {**mapping_chat, **mapping_base,**mapping_extra}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",default="llama-2-7b-chat")
    parser.add_argument("--preference_way",default="binary_choice")
    parser.add_argument("--ask_AB_prompt",default="naive")
    parser.add_argument("--change_AB",default=False, action="store_true")
    parser.add_argument("--end_pos",type=int,default=1000000000,)
    args = parser.parse_args()

    model_name = args.model_name
    assert model_name in mapping
    model_dir = mapping[model_name]["path"]
    wrapper_type = mapping[model_name]["wrapper_type"]
    r1r2_token_ids = mapping[model_name]["AB_token_ids"]

    way = args.preference_way


    vllm_engine = VllmEngine(model_name_or_dir=model_dir)
    # vllm_engine = None
    tokenizer = vllm_engine.llm.get_tokenizer()
    # tokenizer = None

    data = read_all("./shared/data/chatbot_arena_shuffled_no-tie_group_balanced.jsonl")

    if way == "get_prompt_probs":
        all_wrapped_q_rs = []
        prompt_response_tokens = []
        start_pos = len(read_jsonl(f"collected_data/model_preference/{model_name}/query_responses_logprobs.jsonl"))
        for item in data[start_pos:args.end_pos]:
            query = item["prompt"]
            r1 = item["response_a"]
            r2 = item["response_b"]
            w_q1_tokens, r1_tokens, wrapped_q_r1 = wrapper_io(wrapper_type=wrapper_type, query=query, response=r1,
                                                              tokenizer=tokenizer)
            w_q2_tokens, r2_tokens, wrapped_q_r2 = wrapper_io(wrapper_type=wrapper_type, query=query, response=r2,
                                                              tokenizer=tokenizer)
            assert w_q1_tokens == w_q2_tokens
            all_wrapped_q_rs.extend([wrapped_q_r1, wrapped_q_r2])
            prompt_response_tokens.extend([(w_q1_tokens, r1_tokens), (w_q2_tokens, r2_tokens)])

        prompt_logprobs = vllm_engine.get_prompt_token_probs(all_wrapped_q_rs, prompt_response_tokens)

        assert len(prompt_logprobs) == len(data) * 2

        transformed_logprobs = []
        for i in range(len(data)):
            transformed_logprobs.append(
                {"response_a": prompt_logprobs[i * 2], "response_b": prompt_logprobs[i * 2 + 1]})

        write_jsonl(transformed_logprobs,
                    f"collected_data/model_preference/{model_name}/query_responses_logprobs.jsonl",mode="a")
    elif way == "binary_choice":
        all_wrapped_q_rpairs = []
        AB_flag = "AB" if not args.change_AB else "BA"
        start_pos = len(read_all(f"collected_data/model_preference/{model_name}/direct_ask_{args.ask_AB_prompt}_preferences_{AB_flag}.jsonl"))
        for item in data[start_pos:args.end_pos]:
            query = item["prompt"]
            r1 = item["response_a"]
            r2 = item["response_b"]
            if args.change_AB:
                r1, r2 = r2, r1
            sysmsg, wrapped_p_rpair = wrapper_p_rpair(wrapper_type=args.ask_AB_prompt, query=query, response_1=r1, response_2=r2,)
            all_wrapped_q_rpairs.append(wrapped_p_rpair)
            # print(wrapped_p_rpair)
            # raise ValueError

        AB_probs = vllm_engine.get_all_first_generated_token_probs(all_wrapped_q_rpairs, r1r2_token_ids=r1r2_token_ids)
        write_jsonl(AB_probs, f"collected_data/model_preference/{model_name}/direct_ask_{args.ask_AB_prompt}_preferences_{AB_flag}.jsonl",mode="a")
    else:
        raise ValueError(f"way {way} not supported")
