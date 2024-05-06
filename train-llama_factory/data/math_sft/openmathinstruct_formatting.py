import json
import random

def magicoder_convert_format(input_file, output_file, sample_num):
    # Initialize an empty list to store the processed data
    processed_data = []

    # Read the raw data from the jsonl file and sample the specified number of entries
    with open(input_file, 'r') as file:
        lines = file.readlines()
        sampled_lines = random.sample(lines, sample_num)

        for line in sampled_lines:
            raw_entry = json.loads(line)

            # Initialize a dictionary for the processed entry
            processed_entry = {
                "conversations": []
            }

            processed_entry['conversations'].append({
                "from": "user",
                "value": raw_entry["question"]
            })
            processed_entry['conversations'].append({
                "from": "assistant",
                "value": raw_entry["generated_solution"]
            })

            # Add the processed entry to the list
            processed_data.append(processed_entry)

    # Write the processed data to a new json file
    with open(output_file, 'w') as file:
        json.dump(processed_data, file, indent=4)

    print("format conversion complete!")

# Example usage
magicoder_convert_format('open_math_instruct_train.jsonl', 'OpenMathInstruct-1_formatted.json', 50000)
