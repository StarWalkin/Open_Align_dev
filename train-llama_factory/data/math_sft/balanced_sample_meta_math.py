import json

quota = 50000
raw_path = 'meta_math_train.jsonl'

# Type: MATH_AnsAug, Percentage: 18.99%
# Type: GSM_Rephrased, Percentage: 20.25%
# Type: GSM_SV, Percentage: 10.13%
# Type: GSM_FOBAR, Percentage: 10.13%
# Type: GSM_AnsAug, Percentage: 20.25%
# Type: MATH_FOBAR, Percentage: 3.80%
# Type: MATH_Rephrased, Percentage: 12.66%
# Type: MATH_SV, Percentage: 3.80%

type_list = ["MATH_AnsAug", "GSM_Rephrased", "GSM_SV", "GSM_FOBAR", "GSM_AnsAug", "MATH_FOBAR", "MATH_Rephrased", "MATH_SV"]
proportion_list = [0.1899, 0.2025, 0.1013, 0.1013, 0.2025, 0.0380, 0.1266, 0.0380]

# balancedly sample data from raw_path according to proportion_list, with an overall quota of quota
data = []
type_counts = {}
total_count = 0
with open(raw_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        type_name = item['type']
        if type_name in type_list:
            if type_counts.get(type_name, 0) < quota * proportion_list[type_list.index(type_name)]:
                data.append(item)
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
                total_count += 1
            if total_count == quota:
                break

# save the sampled data
with open(f'balanced_sample_meta_math_{quota}.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')
        print(f"file saved to balanced_sample_meta_math_{quota}.jsonl")

