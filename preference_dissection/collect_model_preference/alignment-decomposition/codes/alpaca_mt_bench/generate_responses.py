import random

from shared.codes.utils import *
from shared.codes.vllm_utils import *
from shared.codes.api_utils import *
from codes.utils.utils import *
from shared.codes.prompt_utils import *
from shared.codes.constants import *

from transformers import PreTrainedTokenizer
from typing import Optional, Callable, List, Tuple, Union

import argparse
import time
import shortuuid

ckpt_dir = "/cpfs01/shared/GAIR/GAIR_hdd/ckpts"

openai_api_key = "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"

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

    "alpaca-7b":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ckpts/alpaca/7b","wrapper_type":"alpaca","AB_token_ids":[319, 350]},
    "alpaca-13b": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ckpts/alpaca/13b", "wrapper_type": "alpaca", "AB_token_ids": [319, 350]},

    "alpaca-7b-dpo-positive-chatgpt":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/alpaca_7b_preference/gpt-3.5","wrapper_type": "alpaca", "AB_token_ids":[319, 350]},
    "alpaca-7b-dpo-negative-chatgpt": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/alpaca_7b_preference/gpt-3.5-inv", "wrapper_type": "alpaca", "AB_token_ids": [319, 350]},
    "alpaca-7b-dpo-positive-gpt4":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/alpaca_7b_preference/gpt-4","wrapper_type": "alpaca", "AB_token_ids":[319, 350]},
    "alpaca-7b-dpo-negative-gpt4": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/alpaca_7b_preference/gpt-4-inv", "wrapper_type": "alpaca", "AB_token_ids": [319, 350]},
    "alpaca-7b-dpo-positive-human":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/alpaca_7b_preference/human","wrapper_type": "alpaca", "AB_token_ids":[319, 350]},
    "alpaca-7b-dpo-negative-human": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/alpaca_7b_preference/human-inv", "wrapper_type": "alpaca", "AB_token_ids": [319, 350]},

    "alpaca-13b-dpo-positive-chatgpt":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/alpaca_13b_preference/gpt-3.5","wrapper_type": "alpaca", "AB_token_ids":[319, 350]},
    "alpaca-13b-dpo-negative-chatgpt": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/alpaca_13b_preference/gpt-3.5-inv", "wrapper_type": "alpaca", "AB_token_ids": [319, 350]},
    "alpaca-13b-dpo-positive-gpt4":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/alpaca_13b_preference/gpt-4","wrapper_type": "alpaca", "AB_token_ids":[319, 350]},
    "alpaca-13b-dpo-negative-gpt4": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/alpaca_13b_preference/gpt-4-inv", "wrapper_type": "alpaca", "AB_token_ids": [319, 350]},
    "alpaca-13b-dpo-positive-human":{"path":"/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/alpaca_13b_preference/human","wrapper_type": "alpaca", "AB_token_ids":[319, 350]},
    "alpaca-13b-dpo-negative-human": {"path": "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/alpaca_13b_preference/human-inv", "wrapper_type": "alpaca", "AB_token_ids": [319, 350]},
}


# put mapping_chat and mapping_base together to get mapping
mapping_models = {**mapping_chat, **mapping_base, **mapping_extra}


def build_conv(dialogues, sysmsg=None):
    # we assert user/assistant/user/assistant/... order in the dialogues
    ss = []
    if sysmsg is not None:
        wrapped_sysmsg = {
            "role": "system",
            "content": sysmsg,
        }
        ss.append(wrapped_sysmsg)
    for i, dialogue in enumerate(dialogues):
        wrapped_dialogue = {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": dialogue,
        }
        ss.append(wrapped_dialogue)
    return ss


all_chars = read_yaml("./shared/files/v1.4/all_characteristics.yaml")

IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"


def qwen_make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens

def remove_duplicate(sstr, tokenizer):
    if tokenizer is None:
        return sstr

    if sstr.endswith(IMEND): sstr = sstr[:-len(IMEND)]
    if sstr.endswith(ENDOFTEXT): sstr = sstr[:-len(ENDOFTEXT)]

    def reduce_consecutive_numbers_corrected(numbers):
        """
        Reduces the occurrences of more than 20 consecutive identical integers in a list to 5.
        Args:
        numbers (list): A list of integers.

        Returns:
        list: A new list with reduced consecutive numbers.
        """
        # Edge case: Empty list or list with less than 20 elements
        if len(numbers) < 10:
            return numbers

        # Process the list
        result = []
        count = 1

        for i in range(1, len(numbers)):
            if numbers[i] == numbers[i - 1]:
                count += 1
            else:
                # Add the elements to the result based on the count
                if count > 20:
                    result.extend([numbers[i - 1]] * 5)
                else:
                    result.extend([numbers[i - 1]] * count)
                count = 1

        # Add the last sequence
        if count > 10:
            result.extend([numbers[-1]] * 5)
        else:
            result.extend([numbers[-1]] * count)

        return result

    token_ids = tokenizer.encode(sstr, add_special_tokens=False)
    # if we find 20 more consecutive tokens that are the same, we keep only the first 5
    new_token_ids = reduce_consecutive_numbers_corrected(token_ids)
    return tokenizer.decode(new_token_ids, skip_special_tokens=True)


def build_sysmsg_by_characteristics(chars):
    if len(chars) == 0:
        return None
    ss = "When responding to the user, try your best to make your response satisfy the following characteristics:\n\n"
    ss += "[Characteristics Start]\n"

    for ch in chars:
        ss += f"{ch}: {all_chars[ch]['content']}\n"

    ss += "[Characteristics End]"
    return ss


def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids


model_preference = get_preferences_xx()


def get_preference_by_model(target_judge, preference_select_start=0, preference_select_end=3, scenario_group="Advice",
                            category_wise=True):
    if category_wise:
        preference = model_preference[scenario_group][target_judge]
    else:
        preference = model_preference["all"][target_judge]
    assert preference_select_start < preference_select_end
    assert preference_select_end <= len(preference)
    assert preference_select_start >= 0
    char_idxs = preference[preference_select_start:preference_select_end]
    js = [feature_id_to_name_system_msg[idx] for idx in char_idxs]
    return js

def gen_split_all_to_batch(all_list, batch_size, engine):
    num_batch = math.ceil(len(all_list) / batch_size)
    returned = []
    all_cost = 0.0
    for i in range(num_batch):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(all_list))
        batched_input = all_list[start:end]
        batched_output = engine.generate_batch(batched_input, direct_msg_list=True)
        batch_cost = sum([item.get("cost",0.0) for item in batched_output])
        all_cost += batch_cost
        print(f"batch {i} | cost {batch_cost:.2f} | total cost {all_cost:.2f}")
        returned.extend(batched_output)
    return returned


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama-2-7b-chat')
    parser.add_argument('--dataset', type=str, default="alpaca_eval")
    parser.add_argument('--sys_in_mtbench', type=str, default="1-none", choices=["1-1", "1-2", "1-none"])
    parser.add_argument('--subset_in_alpacaeval',action="store_true")
    parser.add_argument("--target_judge", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--select_way", type=str, default="top", choices=["assign", "random", "none"])
    parser.add_argument("--select_start", type=int, default=0)
    parser.add_argument("--select_end", type=int, default=3)
    parser.add_argument("--category_wise", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ### prepare data
    if args.dataset == "alpaca_eval":
        subset_idxs = [190, 580, 193, 266, 634, 134, 579, 150, 676, 518, 431, 750, 428, 344, 723, 459, 81, 56, 166, 363, 526, 652,
         466, 89, 137, 509, 223, 395, 68, 699, 597, 415, 228, 608, 749, 651, 584, 754, 16, 257, 493, 116, 685, 280, 87,
         63, 5, 297, 38, 334, 616, 581, 80, 97, 688, 524, 434, 666,
         224, 32, 367, 94, 766, 733, 30, 239, 243, 642, 658, 558, 411, 473, 45, 18, 570, 162, 227, 376, 40, 693]

        d1 = read_all("collected_data/alpaca_eval/example/text_davinci_003.json")
        wscenarios = read_all("shared/data/alpacaeval.jsonl")
        assert len(d1) == len(wscenarios)
        instructions = [item['instruction'] for item in d1]
        scenario_groups = [reversed_scenario_group[item['scenario']] for item in wscenarios]

        if args.subset_in_alpacaeval:
            d1 = [d1[i] for i in subset_idxs]
            wscenarios = [wscenarios[i] for i in subset_idxs]
            instructions = [instructions[i] for i in subset_idxs]
            scenario_groups = [scenario_groups[i] for i in subset_idxs]

    elif args.dataset == "mt_bench":
        d1 = read_all("./shared/data/mtbench.json")
        instructions = [item['turns'][0] for item in d1]
        instructions_second = [item['turns'][1] for item in d1]
        scenario_groups = [reversed_scenario_group[item['scenario']] for item in d1]
        scenario_groups_2 = [reversed_scenario_group[item['scenario_2']] for item in d1]
    else:
        raise ValueError

    ### prepare model
    if not args.model_name.startswith('gpt'):
        local_run=True
        batchsize = 1000
        model_dir = mapping_models[args.model_name]["path"]
        wrapper_type = mapping_models[args.model_name]["wrapper_type"]
        tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
        stop_token_ids = [tokenizer.eos_token_id]
        if args.model_name.startswith("qwen"):
            for stop_id in get_stop_words_ids("chatml", tokenizer):
                stop_token_ids.extend(stop_id)

        extra_kwargs = {"dummy_tokenizer":"/cpfs01/shared/GAIR/GAIR_hdd/ckpts/llama-2/llama-dummy-tokenizer"} if wrapper_type=="alpaca" else {}
        engine = VllmEngine(model_name_or_dir=model_dir, temperature=0.0, max_new_tokens=2048, stop_token_ids=stop_token_ids,**extra_kwargs)
    else:
        local_run=False
        batchsize=10
        wrapper_type=None
        tokenizer = None
        engine = OpenAIChat(api_key=openai_api_key, model=args.model_name,
                 temperature=0.0, max_tokens=2048, top_p=1.0,
                 frequency_penalty=0, presence_penalty=0, request_timeout=120,
                 type="text", seed=42, return_logprobs=False)

    random.seed(args.seed)
    args.select_num = args.select_end - args.select_start
    fixed_random_chars = random.sample(list(all_chars.keys()), args.select_num)

    # sysmsg = build_sysmsg_by_characteristics(chars)

    show_limit = 5
    curr_show = 0

    if args.select_way == "assign":
        select_part = f"assign{args.select_start}-{args.select_end}"
    elif args.select_way == "none":
        select_part = "none"
    else:
        select_part = f"random{args.select_num}"
    target_part = f"-target_{args.target_judge}" if args.select_way == "assign" else ""
    seed_part = "" if args.select_way != "random" else f"-seed_{args.seed}"
    category_wise_part = "" if args.category_wise else "-categorywise_False"

    sysmsg_tag = f"{select_part}{seed_part}{target_part}{category_wise_part}"
    generator_tag = f"model_{args.model_name}-sysmsg_{sysmsg_tag}"
    if args.dataset == "mt_bench" and args.select_way=="assign":
        generator_tag += f"_mtbench-{args.sys_in_mtbench}"
    if args.dataset == "alpaca_eval" and args.subset_in_alpacaeval:
        sysmsg_tag += "_subset"
    print_colored_text(f">>>>>>>>>>>>>>>>>> {generator_tag} >>>>>>>>>>>>>>>>>>", "yellow")
    conversations = []
    for item, scenario_group in zip(instructions, scenario_groups):
        if args.select_way == "random":
            chars = fixed_random_chars
        elif args.select_way == "none":
            chars = []
        else:
            chars = get_preference_by_model(args.target_judge, preference_select_start=args.select_start,
                                            preference_select_end=args.select_end, scenario_group=scenario_group,
                                            category_wise=args.category_wise)
        if curr_show < show_limit:
            print(f"{args.select_way}: {args.select_start}-{args.select_end} | {scenario_group} : {chars}")
            curr_show += 1
        sysmsg = build_sysmsg_by_characteristics(chars)
        conversations.append(build_conv([item], sysmsg=sysmsg))
    if local_run:
        wrapped_instructions = [wrapper_input(wrapper_type, tokenizer, conversation) for conversation in conversations]
    else:
        wrapped_instructions = conversations
    print_colored_text("Show the first 5 full inputs ...", "yellow")
    for i in range(5):
        print(wrapped_instructions[i])
        print_colored_text("--------------------------------------------------", "yellow")
    print_colored_text("Start generating the first turn ...", "cyan")
    # outputs = engine.generate_batch(wrapped_instructions,direct_msg_list=True)
    outputs = gen_split_all_to_batch(wrapped_instructions, batchsize, engine)

    show_limit = 5
    curr_show = 0

    if args.dataset == "alpaca_eval":
        assert len(outputs) == len(d1)
        for i, item in enumerate(d1):
            d1[i]['output'] = remove_duplicate(outputs[i]['output'].strip(), tokenizer)
            d1[i]['generator'] = generator_tag

        write_json(d1, f"collected_data/alpaca_eval/model_inference/{args.model_name}/sysmsg_{sysmsg_tag}.json")
    else:
        # we need to generate for the second turn
        model_turn1_responses = [remove_duplicate(item['output'].strip(), tokenizer) for item in outputs]
        assert len(instructions) == len(model_turn1_responses) == len(instructions_second)
        conversations = []
        for idx, (ut1, mt1, ut2) in enumerate(zip(instructions, model_turn1_responses, instructions_second)):
            scenario_group = scenario_groups[idx]
            scenario_group_2 = scenario_groups_2[idx]
            if args.select_way == "none" or args.sys_in_mtbench == "1-none":
                chars = []
            elif args.select_way == "random":
                chars = fixed_random_chars
            else:
                if args.sys_in_mtbench == "1-1":
                    chosen_scenario_group = scenario_group
                elif args.sys_in_mtbench == "1-2":
                    chosen_scenario_group = scenario_group_2
                else:
                    raise ValueError
                chars = get_preference_by_model(args.target_judge, preference_select_start=args.select_start,
                                            preference_select_end=args.select_end, scenario_group=scenario_group,
                                            category_wise=args.category_wise)
            if curr_show < show_limit:
                print(f"{args.select_way}-{args.select_start} | {scenario_group} : {chars}")
                curr_show += 1
            sysmsg = build_sysmsg_by_characteristics(chars)
            conversations.append(build_conv([ut1, mt1, ut2], sysmsg=sysmsg))
        if local_run:
            wrapped_instructions = [wrapper_input(wrapper_type, tokenizer, conversation) for conversation in
                                    conversations]
        else:
            wrapped_instructions = conversations
        print_colored_text("Start generating the second turn ...", "cyan")
        # outputs = engine.generate_batch(wrapped_instructions, direct_msg_list=True)
        outputs = gen_split_all_to_batch(wrapped_instructions, batchsize, engine)
        model_turn2_responses = [item['output'].strip() for item in outputs]
        assert len(instructions) == len(model_turn1_responses) == len(instructions_second) == len(model_turn2_responses)
        final_outputs = []
        for i, (mt1, mt2) in enumerate(zip(model_turn1_responses, model_turn2_responses)):
            final_outputs.append({
                "question_id": 81 + i,
                "answer_id": shortuuid.uuid(),
                "model_id": generator_tag,
                "choices": [
                    {"index": 0, "turns": [remove_duplicate(mt1, tokenizer), remove_duplicate(mt2, tokenizer)]}
                ],
                "tstamp": time.time()
            })
        write_jsonl(final_outputs,
                    f"codes/alpaca_mt_bench/mt_bench_all/data/mt_bench/model_answer/{generator_tag}.jsonl")
