import json
import copy
from shared.codes.utils import *
from shared.codes.api_utils import *

import math


def calculate_pie():
    # Calculate the area of a pie
    area = math.pi * (0.5 * (1.0 - 0.5))
    return area

if __name__ == '__main__':
    datas = read_all("./shared/data/chatbot_arena_shuffled_no-tie.jsonl")
    grammars = []
    for data in tqdm(datas):
        r1_grammar_errors = grammar_check(data['response_a'])
        r2_grammar_errors = grammar_check(data['response_b'])
        data['r1_num_grammar_errors'] = r1_grammar_errors[1]
        data['r2_num_grammar_errors'] = r2_grammar_errors[1]
        xx = {"Response 1": r1_grammar_errors[0], "Response 2": r2_grammar_errors[0]}
        grammars.append(xx)

    write_jsonl(datas, "./shared/data/chatbot_arena_shuffled_no-tie_annotated.jsonl")
    write_jsonl(grammars, "./shared/data/chatbot_arena_shuffled_no-tie_full-grammar-errors.jsonl")

