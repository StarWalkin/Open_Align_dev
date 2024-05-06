from datasets import load_dataset
import json

def dataset_to_jsonl(dataset_name, subset_name, split, output_file):
    # Adjusted to include a subset name
    dataset = load_dataset(dataset_name, subset_name, split=split)

    with open(output_file, 'w', encoding='utf-8') as file:  # Added encoding for better compatibility
        for item in dataset:
            json_str = json.dumps(item)
            file.write(json_str + '\n')

    print(f'Dataset {dataset_name}/{subset_name} {split} split has been converted to {output_file}')

# Now you call the function with the subset name as an additional argument
dataset_to_jsonl('gsm8k', 'main', 'train', 'gsm8k_train.jsonl')
