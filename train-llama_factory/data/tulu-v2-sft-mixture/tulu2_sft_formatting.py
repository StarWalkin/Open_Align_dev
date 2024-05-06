import json


def tulu2_convert_format(input_file, output_file):
    # Initialize an empty list to store the processed data
    processed_data = []

    # Read the raw data from the jsonl file
    with open(input_file, 'r') as file:
        for line in file:
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

    # Write the processed data to a new json file
    with open(output_file, 'w') as file:
        json.dump(processed_data, file, indent=4)

    print("format conversion complete!")

# Example usage
tulu2_convert_format('tulu_v2_sft_mixture_train.jsonl', 'tulu2_lf.json')