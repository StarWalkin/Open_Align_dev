import json
from utils import *


model_name = "gemma-2b-it"

pref1 = read_all(f"/cpfs01/user/liupengfei/yguo/Open_Align_dev/preference_dissection/{model_name}/direct_ask_naive_preferences_AB.jsonl")
pref2 = read_all(f"/cpfs01/user/liupengfei/yguo/Open_Align_dev/preference_dissection/{model_name}/direct_ask_naive_preferences_BA.jsonl")

choice_res = []
for i in range(len(pref1)):
    prob_a = pref1[i]["response_a"] + pref2[i]["response_b"]
    prob_b = pref1[i]["response_b"] + pref2[i]["response_a"]
    if prob_a > prob_b:
        choice_res.append("A")
    else:
        choice_res.append("B")


save_path = "gemma-2b-it_choice.json"
# write_json(choice_res, save_path)

with open(save_path, "w") as f:
    json.dump(choice_res, f)

print(f"Successfully written to {save_path}")