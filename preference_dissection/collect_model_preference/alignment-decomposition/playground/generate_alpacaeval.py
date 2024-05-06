import random

from shared.codes.utils import *
from shared.codes.vllm_utils import *
from shared.codes.api_utils import *
from shared.codes.prompt_utils import *

import argparse

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
mapping = {**mapping_chat, **mapping_base}


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


def build_sysmsg_by_characteristics(chars):
    ss = "When responding to the user, try your best to make your response satisfy the following characteristics:\n\n"
    ss+="[Characteristics Start]\n"

    for ch in chars:
        ss += f"{ch}: {all_chars[ch]['content']}\n"

    ss+="[Characteristics End]"
    return ss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='llama-2-7b-chat')
    args = parser.parse_args()

    d1 = read_all("collected_data/alpaca_eval/example/text_davinci_003.json")
    instructions = [item['instruction'] for item in d1]
    model_dir = mapping[args.model_name]["path"]
    wrapper_type = mapping[args.model_name]["wrapper_type"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    vllm_engine = VllmEngine(model_name_or_dir=model_dir, temperature=0.0, max_new_tokens=2048, )
    # vllm_engine = None

    top5_chars = ["innovative and novel", "information richness without considering inaccuracy", "lengthy",
                  "relevance without considering inaccuracy", "friendly"]
    bottom5_chars = ["admit limitations or mistakes", "repetitive", "funny and humorous", "polite",
                     "use of informal expressions"]
    random5_chars = ['correcting explicit mistakes or biases', 'authoritative tone', 'repetitive', 'lengthy',
                     'interactive']
    mapping = {"top-5": top5_chars, "bottom-5": bottom5_chars, "random-5": random5_chars}

    for sysmsg_tag, chars in mapping.items():

        sysmsg = build_sysmsg_by_characteristics(chars)

        # sysmsg_tag = "none"
        generator_tag = f"model_{args.model_name}-sysmsg_{sysmsg_tag}"
        #

        conversations = [build_conv([item], sysmsg=sysmsg) for item in instructions]

        wrapped_instructions = [wrapper_input(wrapper_type, tokenizer, conversation) for conversation in conversations]

        outputs = vllm_engine.generate_batch(wrapped_instructions)

        # write_jsonl(outputs,f"collected_data/alpaca_eval/model_inference/{args.model_name}/sysmsg_{sysmsg}.jsonl")
        #
        # outputs = read_all(f"collected_data/alpaca_eval/model_inference/{args.model_name}/sysmsg_{sysmsg}.jsonl")

        assert len(outputs) == len(d1)
        for i, item in enumerate(d1):
            d1[i]['output'] = outputs[i]['output'].lstrip()
            d1[i]['generator'] = generator_tag

        write_json(d1, f"collected_data/alpaca_eval/model_inference/{args.model_name}/sysmsg_{sysmsg_tag}.json")
