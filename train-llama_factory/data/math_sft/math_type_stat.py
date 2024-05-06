import json

# 读取 JSONL 文件并解析成 Python 对象
data = []
with open("meta_math_train.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

# 统计各个类型的计数
type_counts = {}
total_count = len(data)
for item in data:
    type_name = item["type"]
    type_counts[type_name] = type_counts.get(type_name, 0) + 1

# 计算各个类型的占比
type_percentages = {}
for type_name, count in type_counts.items():
    percentage = (count / total_count) * 100
    type_percentages[type_name] = percentage

# 打印结果
for type_name, percentage in type_percentages.items():
    print(f"Type: {type_name}, Percentage: {percentage:.2f}%")
