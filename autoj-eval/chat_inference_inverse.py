from vllm import LLM, SamplingParams
import torch
from constants_prompt import build_autoj_input
import json
import os


def write_jsonl(data, jsonl_file_path, mode="w"):
    # data is a list, each of the item is json-serilizable
    assert isinstance(data, list)
    if not os.path.exists(os.path.dirname(jsonl_file_path)):
       os.makedirs(os.path.dirname(jsonl_file_path))
    with open(jsonl_file_path, mode) as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def extract_pariwise_result(raw_output):
    raw_output = raw_output.strip()
    pos = raw_output.rfind('final decision is ')
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len('final decision is '):].strip().lower()
        if pred_rest.startswith('response 1'):
            pred_label = 0
        elif pred_rest.startswith('response 2'):
            pred_label = 1
        elif pred_rest.startswith('tie'):
            pred_label = 2
    return pred_label


def extract_single_rating(score_output):
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        return float(score_output[pos + len("[["):pos2].strip())
    else:
        return 0.0


def generate_choice(input_data):
     input_data = json.loads(input_data)
     input_pairwise = build_autoj_input(prompt = input_data["prompt"],
                                       resp1 = input_data["response 2"],
                                       resp2 = input_data["response 1"],
                                       protocol="pairwise_tie")
     outputs = llm.generate(input_pairwise, sampling_params)
     judgment = outputs[0].outputs[0].text
     print(judgment)
     evaluation_result = extract_pariwise_result(judgment)
     single_res = {"output": evaluation_result}
     return single_res


if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    # model_name_or_dir = "/cpfs01/shared/GAIR/GAIR_hdd/jlli/llama2-sft/13b/autoj/ood-nonlptasks/575-converted"  # or "local path to auto-j"
    # model_name_or_dir = "/cpfs01/user/liupengfei/yguo/ckpts/autoj-sft7-yi-6B-final"  # or "local path to auto-j"
    model_name_or_dir = "/cpfs01/shared/GAIR/GAIR_hdd/yguo/ckpts/Qwen/Qwen1.5-7B-Chat"  # or "local path to auto-j"
    llm = LLM(model=model_name_or_dir, tensor_parallel_size=num_gpus)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)

    with open("/cpfs01/user/liupengfei/yguo/zn_autoj/zn_autoj_eval/evaluation/english_test/testdata_pairwise.jsonl") as file:
        pairwise_testdata = file.readlines()
    
    res_list = []
    for data in pairwise_testdata:
        res_list.append(generate_choice(data))

    print("finished generation")

    write_jsonl(res_list, "./en_pairwise_outputs_chat_exchange.jsonl", "a")

    print("completed output!")
