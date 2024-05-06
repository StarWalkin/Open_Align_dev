import json
import random

def meta_math_convert_format(input_file, output_file):
    # Initialize an empty list to store the processed data
    processed_data = []

    # Read the raw data from the jsonl file and sample the specified number of entries
    with open(input_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
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
    with open(output_file, 'w') as file:
        json.dump(processed_data, file, indent=4)

    print("format conversion complete!")

# Example usage
meta_math_convert_format('meta_math_train.jsonl', 'meta_math_full.json')
