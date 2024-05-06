# import json
# import random

# def magicoder_convert_format(input_file, output_file, sample_num):
#     # Initialize an empty list to store the processed data
#     processed_data = []

#     # Read the raw data from the jsonl file and sample the specified number of entries
#     with open(input_file, 'r') as file:
#         lines = file.readlines()
#         sampled_lines = random.sample(lines, sample_num)

#         for line in sampled_lines:
#             raw_entry = json.loads(line)

#             # Initialize a dictionary for the processed entry
#             processed_entry = {
#                 "conversations": []
#             }

#             processed_entry['conversations'].append({
#                 "from": "user",
#                 "value": raw_entry["instruction"]
#             })
#             processed_entry['conversations'].append({
#                 "from": "assistant",
#                 "value": raw_entry["response"]
#             })

#             # Add the processed entry to the list
#             processed_data.append(processed_entry)

#     # Write the processed data to a new json file
#     with open(output_file, 'w') as file:
#         json.dump(processed_data, file, indent=4)

#     print("format conversion complete!")

# # Example usage
# magicoder_convert_format('Magicoder-Evol-Instruct-110K_train.jsonl', 'magicoder_formatted.json', 50000)


import json
import random

def magicoder_convert_format(input_file, output_file):
    # Initialize an empty list to store the processed data
    processed_data = []

    # Read the raw data from the jsonl file and sample the specified number of entries
    with open(input_file, 'r') as file:
        lines = file.readlines()

        # # randomly choose 50k data out of all the data
        # lines = random.sample(lines, 50000)


        for line in lines:
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

    # Write the processed data to a new json file
    with open(output_file, 'w') as file:
        json.dump(processed_data, file, indent=4)

    print("format conversion complete!")


def magicoder_sample_and_convert_format(input_file, output_file):
    # Initialize an empty list to store the processed data
    processed_data = []

    # Read the raw data from the jsonl file and sample the specified number of entries
    with open(input_file, 'r') as file:
        lines = file.readlines()

        # randomly choose 50k data out of all the data
        lines = random.sample(lines, 50000)


        for line in lines:
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

    # Write the processed data to a new json file
    with open(output_file, 'w') as file:
        json.dump(processed_data, file, indent=4)

    print("sampling and format conversion complete!")


# Example usage
# magicoder_convert_format('Magicoder-Evol-Instruct-110K_train.jsonl', 'magicoder_formatted_full.json')

magicoder_convert_format('Magicoder-Evol-Instruct-110K_train.jsonl', 'magicoder_formatted_50k.json')
