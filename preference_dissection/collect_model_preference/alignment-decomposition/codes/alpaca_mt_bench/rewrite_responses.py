import random

from shared.codes.utils import *
from shared.codes.vllm_utils import *
from shared.codes.api_utils import *
from shared.codes.prompt_utils import *
from codes.utils.utils import *
from shared.codes.constants import *


import argparse
import time
import shortuuid

ckpt_dir = "/cpfs01/shared/GAIR/GAIR_hdd/ckpts"
mapping_chat = {
    "llama-2-7b-chat": {"path": f"{ckpt_dir}/llama-2/7b-chat", "wrapper_type": "llama-2-chat",
                        "AB_token_ids": [319, 350]},
    "llama-2-13b-chat": {"path": f"{ckpt_dir}/llama-2/llama-2-13b-chat-hf", "wrapper_type": "llama-2-chat",
                         "AB_token_ids": [319, 350]},
    "llama-2-70b-chat": {"path": f"{ckpt_dir}/llama-2/llama-2-70b-chat-hf", "wrapper_type": "llama-2-chat",
                         "AB_token_ids": [319, 350]},

    "vicuna-7b-v1.5": {"path": f"{ckpt_dir}/vicuna/7b-v1.5", "wrapper_type": "vicuna", "AB_token_ids": [319, 350]},
    "vicuna-13b-v1.5": {"path": f"{ckpt_dir}/vicuna/13b-v1.5", "wrapper_type": "vicuna", "AB_token_ids": [319, 350]},
    "wizardLM-13b-v1.2": {"path": f"{ckpt_dir}/WizardLM/WizardLM-13B-V1.2", "wrapper_type": "vicuna",
                          "AB_token_ids": [319, 350]},
    "wizardLM-70b-v1.0": {"path": f"{ckpt_dir}/WizardLM/WizardLM-70B-V1.0", "wrapper_type": "vicuna",
                          "AB_token_ids": [319, 350]},

    "tulu-2-dpo-7b": {"path": f"{ckpt_dir}/tulu/2-dpo-7b", "wrapper_type": "tulu", "AB_token_ids": [319, 350]},
    "tulu-2-dpo-13b": {"path": f"{ckpt_dir}/tulu/2-dpo-13b", "wrapper_type": "tulu", "AB_token_ids": [319, 350]},
    "tulu-2-dpo-70b": {"path": f"{ckpt_dir}/tulu/2-dpo-70b", "wrapper_type": "tulu", "AB_token_ids": [319, 350]},

    "mistral-7b-instruct-v0.1": {"path": f"{ckpt_dir}/mistralai/Mistral-7B-Instruct-v0.1", "wrapper_type": "mistral",
                                 "AB_token_ids": [330, 365]},
    "mistral-7b-instruct-v0.2": {"path": f"{ckpt_dir}/mistralai/Mistral-7B-Instruct-v0.2", "wrapper_type": "mistral",
                                 "AB_token_ids": [330, 365]},
    "mistral-8x7b-instruct-v0.1": {"path": f"{ckpt_dir}/mistralai/mixtral-8x7b-instruct-v0.1",
                                   "wrapper_type": "mistral", "AB_token_ids": [330, 365]},

    "yi-6b-chat": {"path": f"{ckpt_dir}/yi/Yi-6B-Chat", "wrapper_type": "yi", "AB_token_ids": [647, 690]},
    "yi-34b-chat": {"path": f"{ckpt_dir}/yi/Yi-34B-Chat", "wrapper_type": "yi", "AB_token_ids": [647, 690]},

    "zephyr-7b-beta": {"path": f"{ckpt_dir}/zephyr/zephyr-7b-beta", "wrapper_type": "zephyr",
                       "AB_token_ids": [330, 365]},
}

mapping_base = {
    "llama-2-7b": {"path": f"{ckpt_dir}/llama-2/7b", "wrapper_type": "llama-2", "AB_token_ids": [319, 350]},
    "llama-2-13b": {"path": f"{ckpt_dir}/llama-2/13b", "wrapper_type": "llama-2", "AB_token_ids": [319, 350]},
    "llama-2-70b": {"path": f"{ckpt_dir}/llama-2/70b", "wrapper_type": "llama-2", "AB_token_ids": [319, 350]},
    "mistral-7b": {"path": f"{ckpt_dir}/mistralai/Mistral-7B-v0.1", "wrapper_type": "mistral",
                   "AB_token_ids": [330, 365]},
    "mistral-8x7b": {"path": f"{ckpt_dir}/mistralai/Mixtral-8x7B-v0.1", "wrapper_type": "mistral",
                     "AB_token_ids": [330, 365]},
    "yi-6b": {"path": f"{ckpt_dir}/yi/Yi-6B-Llama", "wrapper_type": "yi", "AB_token_ids": [647, 690]},
    "yi-34b": {"path": f"{ckpt_dir}/yi/Yi-34B-Llama", "wrapper_type": "yi", "AB_token_ids": [647, 690]},
}
# put mapping_chat and mapping_base together to get mapping
mapping_models = {**mapping_chat, **mapping_base}


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

def remove_duplicate(sstr, tokenizer):
    def reduce_consecutive_numbers_corrected(numbers):
        """
        Reduces the occurrences of more than 20 consecutive identical integers in a list to 5.
        Args:
        numbers (list): A list of integers.

        Returns:
        list: A new list with reduced consecutive numbers.
        """
        # Edge case: Empty list or list with less than 20 elements
        if len(numbers) < 20:
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
        if count > 20:
            result.extend([numbers[-1]] * 5)
        else:
            result.extend([numbers[-1]] * count)

        return result
    token_ids = tokenizer.encode(sstr,add_special_tokens=False)
    # if we find 20 more consecutive tokens that are the same, we keep only the first 5
    new_token_ids = reduce_consecutive_numbers_corrected(token_ids)
    return tokenizer.decode(new_token_ids,skip_special_tokens=True)


def remove_response_tags(text):
    pattern = r'\[.*?Response (Start|End).*?\]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def remove_query_tags(text):
    pattern = r'\[.*?Query (Start|End).*?\]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def remove_nonsense(sstr):
    sstr = sstr.strip()
    # if the first line of `sstr` contains `rewrite` or `rewritten`, then we remove the first line
    # if the last line of `sstr` contains `rewrite` or `rewritten`, then we remove the last line

    first_n = sstr.find("\n")
    if first_n != -1:
        first_line = sstr[:first_n]
        if "rewrite" in first_line or "rewritten" in first_line:
            sstr = sstr[first_n + 1:]

    sstr = sstr.strip()

    last_n = sstr.rfind("\n")
    if last_n != -1:
        last_line = sstr[last_n + 1:]
        if "rewrite" in last_line or "rewritten" in last_line:
            sstr = sstr[:last_n]

    sstr = remove_query_tags(sstr)
    sstr = remove_response_tags(sstr)

    return sstr.strip()



def build_target_characteristics(chars):
    assert len(chars) > 0
    ss = ""
    ss += "[Characteristics Start]\n"

    for ch in chars:
        ss += f"{ch}: {all_chars[ch]['content']}\n"

    ss += "[Characteristics End]"
    return ss


template = """You will be given a user's query and a response generated by an AI assistant. Your task is to rewrite the response, while ensuring that the original content remains largely unchanged, to better meet the following characteristics.:

{target_chars_str}

Here are the user's query and the response generated by the AI assistant:

[Query Start]
{query}
[Query End]

[Response Start]
{response}
[Response End]

Please output the rewritten response directly without any other information.
"""

model_preference = get_preferences_xx()

def get_preference_by_model(target_judge, preference_select_way="top",scenario_group="Advice",category_wise=True):
    if category_wise:
        preference = model_preference[scenario_group][target_judge]
    else:
        preference = model_preference["all"][target_judge]
    if preference_select_way=="top":
        char_idxs = preference[:args.select_num]
    elif preference_select_way=="bottom":
        char_idxs = preference[-args.select_num:][::-1]
    else:
        raise ValueError
    js = [feature_id_to_name_system_msg[idx] for idx in char_idxs]
    return js



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama-2-7b-chat')
    parser.add_argument('--dataset', type=str, default="alpaca_eval")
    parser.add_argument("--target_judge", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--select_way", type=str, default="top", choices=["top", "bottom", "random"])
    parser.add_argument("--select_num", type=int, default=5)
    parser.add_argument("--category_wise", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert args.dataset == "alpaca_eval", "We now only support alpaca_eval dataset for rewriting"
    raw_inference_file = f"collected_data/alpaca_eval/model_inference/{args.model_name}/sysmsg_none.json"

    assert os.path.exists(raw_inference_file), f"raw_inference_file {raw_inference_file} does not exist"

    # load raw inference file
    raw_inference = read_all(raw_inference_file)

    wscenarios = read_all("shared/data/alpacaeval.jsonl")
    assert len(raw_inference) == len(wscenarios)
    scenario_groups = [reversed_scenario_group[item['scenario']] for item in wscenarios]


    ### prepare model
    model_dir = mapping_models[args.model_name]["path"]
    wrapper_type = mapping_models[args.model_name]["wrapper_type"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    vllm_engine = VllmEngine(model_name_or_dir=model_dir, temperature=0.0, max_new_tokens=2048, )
    # vllm_engine = None

    random.seed(args.seed)
    fixed_random_chars = random.sample(list(all_chars.keys()), args.select_num)

    generator_tag = f"{args.select_way}{args.select_num}-seed_{args.seed}-target_{args.target_judge}-categorywise_{args.category_wise}"
    print_colored_text(f">>>>>>>>>>>>>>>>>> {generator_tag} >>>>>>>>>>>>>>>>>>", "yellow")


    rewrite_prompts = []
    for item,scenario_group in zip(raw_inference,scenario_groups):
        if args.select_way == "random":
            chars = fixed_random_chars
        elif args.select_way == "none":
            chars = []
        else:
            chars = get_preference_by_model(args.target_judge, preference_select_way=args.select_way,
                                            scenario_group=scenario_group, category_wise=args.category_wise)
        target_chars_str = build_target_characteristics(chars)
        rewrite_prompts.append(template.format(target_chars_str=target_chars_str, query=item['instruction'], response=item['output']))

    conversations = [build_conv([item]) for item in rewrite_prompts]
    wrapped_instructions = [wrapper_input(wrapper_type, tokenizer, conversation) for conversation in conversations]
    print(f">>>>Show the first wrapped instruction: \n{wrapped_instructions[0]}\n>>>>")
    sleep(1)
    print_colored_text("Start rewriting ...", "cyan")
    outputs = vllm_engine.generate_batch(wrapped_instructions)

    assert len(outputs) == len(raw_inference)
    for i, item in enumerate(raw_inference):
        raw_inference[i]['output'] = remove_nonsense(remove_duplicate(outputs[i]['output'].strip(),tokenizer))
        raw_inference[i]['generator'] = generator_tag

    write_json(raw_inference, f"collected_data/alpaca_eval/model_inference/{args.model_name}/rewrite_{generator_tag}.json")


