import json
import random

if __name__ == "__main__":
    # read tulu2 data and exclude those from codealpaca
    with open("../tulu-v2-sft-mixture/tulu_v2_sft_mixture_train.jsonl", "r") as f:
        tulu2_lines = f.readlines()
    
    print(len(tulu2_lines))
    print(tulu2_lines[-1])
    cnt = 0
    cnt_all = 0
    # for idx, line in enumerate(tulu2_lines):
    #     cnt_all += 1
    #     raw_entry = json.loads(line)
    #     if raw_entry['dataset'] == "code_alpaca":
    #         # tulu2_lines.remove(line)
    #         tulu2_lines.pop(idx)
    #         cnt += 1
    
    # print(idx)
    # print(f"Excluded {cnt} data from code_alpaca in tulu2 data")
    # print(f"Total number of tulu2 data: {cnt_all}")
    tulu2_lines = [line for line in tulu2_lines if json.loads(line)['dataset'] != "code_alpaca"]

    # read magicoder data and randomly sample 50k data
    with open("../magicoder-110k/Magicoder-Evol-Instruct-110K_train.jsonl", "r") as f:
        magicoder_lines = f.readlines()
        magicoder_lines = random.sample(magicoder_lines, 50000)
    
    # read the randomly sampled math data
    with open("../math_sft/balanced_sample_meta_math_50000.jsonl", "r") as f:
        math_lines = f.readlines()
    
    # merge the three datasets and convert to the target json format
    processed_data = []
    # process tulu2 data
    for line in tulu2_lines:
        raw_entry = json.loads(line)

        # Initialize a dictionary for the processed entry
        processed_entry = {
            "conversations": [],
            "system": None,
        }

        # Extract and add the system message if it exists
        for message in raw_entry['messages']:
            if message['role'] == 'system':
                processed_entry['system'] = message['content']

        # Extract and add the user and assistant messages
        for message in raw_entry['messages']:
            if message['role'] in ['user', 'assistant']:
                processed_entry['conversations'].append({
                    "from": message['role'],
                    "value": message['content']
                })

        # Add the processed entry to the list
        processed_data.append(processed_entry)

    # process magicoder data
    for line in magicoder_lines:
        raw_entry = json.loads(line)

        # Initialize a dictionary for the processed entry
        processed_entry = {
            "conversations": []
        }

        processed_entry['conversations'].append({
            "from": "user",
            "value": raw_entry["instruction"]
        })
        processed_entry['conversations'].append({
            "from": "assistant",
            "value": raw_entry["response"]
        })

        # Add the processed entry to the list
        processed_data.append(processed_entry)

    # process metamath data
    for line in math_lines:
        raw_entry = json.loads(line)

        # Initialize a dictionary for the processed entry
        processed_entry = {
            "conversations": []
        }

        processed_entry['conversations'].append({
            "from": "user",
            "value": raw_entry["query"]
        })
        processed_entry['conversations'].append({
            "from": "assistant",
            "value": raw_entry["response"]
        })

        # Add the processed entry to the list
        processed_data.append(processed_entry)

    # Write the processed data to a new json file
    with open('tulu2_math_code_enhanced.json', 'w') as file:
        json.dump(processed_data, file, indent=4)

    print("selection and merge procedure complete! Data successfully written to 'tulu2_math_code_enhanced.json'!")
