from datasets import load_dataset
import json

def dataset_to_jsonl(dataset_name, split, output_file):
    dataset = load_dataset(dataset_name, split=split)

    with open(output_file, 'w') as file:
        for item in dataset:
            json_str = json.dumps(item)
            file.write(json_str + '\n')

    print(f'Dataset {dataset_name} has been converted to {output_file}')

dataset_to_jsonl('allenai/tulu-v2-sft-mixture', 'train', 'tulu_v2_sft_mixture_train.jsonl')