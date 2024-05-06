import numpy as np
import yaml
import openai
from pprint import pprint
import subprocess
import json
import os
import ast
import pdb
import asyncio
from typing import Any, List, Union
import os
import pathlib
from tqdm import tqdm
import tiktoken
from yamlinclude import YamlIncludeConstructor
import pickle
import random
import pandas as pd
import re
import string
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from nltk.tokenize import sent_tokenize, word_tokenize
grammarcktool = None
# try:
#     import language_tool_python
#     grammarcktool = language_tool_python.LanguageTool('en-US')
# except:
#     grammarcktool = None

YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)

programming_languages = ['Python', 'Java', 'C++', 'Cpp', 'C#', 'Csharp', 'JavaScript', 'js', 'Ruby', 'Swift', 'Kotlin',
                         'PHP', 'Go', 'Golong', 'Rust', 'TypeScript', ' R ', 'MATLAB', 'Perl', 'Dart', 'Lua', 'Scala',
                         'Groovy', 'Bash', 'PowerShell', 'Objective-C', 'F#', 'Haskell', 'Julia', 'Ada', 'COBOL',
                         'Fortran', 'Lisp', 'Prolog', 'Scheme', 'Tcl', 'Erlang', 'Clojure', 'Elixir', 'Crystal',
                         'Smalltalk', 'Hack', 'Delphi', 'Pascal', 'Basic', 'PL', 'SQL', 'Scratch', 'Alice', 'Logo',
                         'RPG', 'VHDL', 'Verilog', 'MIPS', 'x86', 'ARM', 'CUDA', 'OpenCL', 'OpenGL', 'Native', 'SNOBOL',
                         'Processing', 'ABAP', 'JCL', 'Vala', 'Etoys', 'Ceylon', 'Solidity', 'Elm', 'Opa',
                         'HaXe']


def safe_write_jsonl(data, jsonl_file_path, mode="w"):
    # check if the directory exists, if not we create it
    if not os.path.exists(os.path.dirname(jsonl_file_path)):
        os.makedirs(os.path.dirname(jsonl_file_path))
    write_jsonl(data, jsonl_file_path, mode)


def read_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def read_jsonl(jsonl_file_path):
    s = []
    if not os.path.exists(jsonl_file_path):
        print_colored_text("File not exists: " + jsonl_file_path, "red")
        return s
    with open(jsonl_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        linex = line.strip()
        if linex == '':
            continue
        s.append(json.loads(linex))
    return s


def read_json(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data


def read_all(file_path):
    if file_path.endswith('.jsonl'):
        return read_jsonl(file_path)
    elif file_path.endswith('.json'):
        return read_json(file_path)
    elif file_path.endswith('.yaml'):
        return read_yaml(file_path)
    elif file_path.endswith('.pkl'):
        return load_pkl(file_path)
    elif file_path.endswith('.parquet'):
        return read_parquet(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip() != '']
    else:
        raise ValueError(f'Unrecognized file type: {file_path}')


def read_parquet(parquet_file_path):
    data = pd.read_parquet(parquet_file_path)
    data = data.to_dict('records')
    return data


def write_jsonl(data, jsonl_file_path, mode="w"):
    # data is a list, each of the item is json-serilizable
    assert isinstance(data, list)
    if not os.path.exists(os.path.dirname(jsonl_file_path)):
        os.makedirs(os.path.dirname(jsonl_file_path))
    with open(jsonl_file_path, mode) as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def write_json(data, json_file_path):
    if not os.path.exists(os.path.dirname(json_file_path)):
        os.makedirs(os.path.dirname(json_file_path))
    with open(json_file_path, 'w') as f:
        json.dump(data, f)


def elegant_show(something, level=0, sid=0, full=False):
    # str,float,int
    # all print in this call should add level*4 spaces
    prefix = "\t" * level

    if isinstance(something, (str, float, int)) or something is None:
        if isinstance(something, str):
            # if '\n' in something:
            #     something = '\n'+something
            # add prefix whenever go to a new line in this string
            something = something.replace("\n", f"\n{prefix}")
        print(prefix, f"\033[1;35mElement: \033[0m", something)
    elif isinstance(something, list) or isinstance(something, tuple) or isinstance(something, np.ndarray):
        # take a random example, and length
        # sid = 0
        if len(something) == 0:
            print(prefix,
                  f"\033[1;33mLen: \033[0m{len(something)} \t\033[1;33m& No elements! \033[0m")
        elif not full or len(something) == 1:
            print(prefix,
                  f"\033[1;33mLen: \033[0m{len(something)} \t\033[1;33m& first element ...\033[0m")
            elegant_show(something[sid], level + 1, sid, full)
        else:
            print(prefix,
                  f"\033[1;33mLen: \033[0m{len(something)} \t\033[1;33m& Elements ...\033[0m")
            for i in range(len(something) - 1):
                elegant_show(something[i], level + 1, sid, full)
                print(prefix + '\t', f"\033[1;33m-------------------------------\033[0m")
            elegant_show(something[-1], level + 1, sid, full)

    elif isinstance(something, dict):
        for k, v in something.items():
            print(prefix, f"\033[1;34mKey: \033[0m{k} \033[1;34m...\033[0m")
            elegant_show(v, level + 1, sid, full)
    else:
        print(prefix, f"\033[1;31mError @ Type: \033[0m{type(something)}")
        raise NotImplementedError


def print_colored_text(text, color="yellow", end=None):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }

    color_code = colors.get(color.lower(), colors["reset"])
    print(f"{color_code}{text}{colors['reset']}", end=end)


def randomly_pick_and_show(items):
    assert type(items) == list
    pools = [i for i in range(len(items))]
    while True:
        if input("q to exit:") == 'q': break
        idx = random.choice(pools)
        pools.pop(idx)
        print_colored_text("idx: " + str(idx), 'red')
        elegant_show(items[idx], full=True)
        print('---------------------------------')


def parquet_to_jsonl(parquet_file_path, jsonl_file_path):
    df = pd.read_parquet(parquet_file_path)
    df.to_json(jsonl_file_path, orient='records', lines=True)


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def resolve_json_like_string(json_like_string):
    """
    resolve json like string to json
    :param json_like_string:
    """
    try:
        return json.loads(json_like_string), True
    except:
        return json_like_string, False


def has_code_snippet(input_string):
    # Regular expressions to detect common code structures
    regex_patterns = [
        r'\b(def|class|if|else|for|while|switch|case|function|async|await|try|catch)\b',  # Keywords
        r'\b(import|from|require|include|using)\b',  # Import statements
        r'(?<!\w)=(?!=)|[(){}\[\];:]',  # Assignment and common code block indicators
        r'\b(return|yield|print|console.log)\b',  # Output statements
        r'(\/\/[^\n]*|\/\*[\s\S]*?\*\/)',  # Single and multi-line comments
    ]

    matches = []
    for pattern in regex_patterns:
        matches.extend(re.findall(pattern, input_string))

    # If there are more than a certain number of matches (you can adjust this threshold),
    # assume the string contains a code snippet
    threshold = 10
    return len(matches) >= threshold


def has_website_link(input_string):
    regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    if re.search(regex, input_string):
        return True
    else:
        return False


def has_math_expr(text):
    # 支持使用括号的算术表达式
    math_expr = re.search("(\d+\s*(\*|\/|\+|\-)\s*)+(\d+\s*)", text)

    if math_expr:
        return True
    else:
        return False


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def heuristic_postprocess_parwise(raw_output):
    pos = raw_output.find("\"explanation\":")
    if pos != -1:
        endpos = pos + len("\"explanation\":")
        # find the first \" and second \" in raw_output[endpos:]
        pos1 = raw_output[endpos:].find("\"") + endpos + 1
        pos2 = raw_output[endpos + pos1 + 1:].find("\"") + endpos + pos1 + 1
        if pos1 != -1 and pos2 != -1:
            # replace all \n to \\n in raw_output[pos1:pos2]
            raw_output = raw_output[:pos1] + raw_output[pos1:pos2].replace('\n', '\\n') + raw_output[pos2:]
    return raw_output



def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def heuristic_filter(scenario, input_text):
    # input the first stage scenario and input_text
    # return the final decision
    if scenario == "code_generation":
        full_list = ['code', 'script', "app", "program"] + programming_languages
        for word in full_list:
            if word.lower() in input_text.lower():
                return "code_generation"
        return "default"
    elif scenario == "asking_how_to_question":
        if "how to" in input_text.lower():
            return "asking_how_to_question"
        return "default"
    elif scenario == "explaining_code":
        if not has_code_snippet(input_text):
            return "explaining_general"
        return "explaining_code"
    elif scenario == "verifying_fact":
        if " or " in input_text.lower():
            return "default"
        return "verifying_fact"
    elif scenario == "text_summarization":
        num_words = len(input_text.split())
        if num_words < 50:
            if has_website_link(input_text):
                return "rejecting"
            return "default"
        cand_list = ["summary", "summarizing", "tldr", "tl;dr", "summarization", "summarize", "summarise"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "text_summarization"
        return "default"
    elif scenario == "writing_blog_post":
        cand_list = ["blog", "blogpost"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "writing_blog_post"
        return "default"
    elif scenario == "code_correction":
        if not has_code_snippet(input_text):
            return "default"
        return "code_correction"
    elif scenario == "text_to_text_translation":
        cand_list = ["translate", "translation", "translator", "翻译"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "text_to_text_translation"
        return "default"
    elif scenario == "writing_email":
        cand_list = ["mail", "email"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "writing_email"
        return "default"
    elif scenario == "question_generation":
        cand_list = ["question"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "question_generation"
        return "default"
    elif scenario == "analyzing_general":
        cand_list = ["analyze", "analysis", "analytical", "discuss", "discussion", "compare", "comparison", "research"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "analyzing_general"
        return "default"
    elif scenario == "code_to_code_translation":
        if not has_code_snippet(input_text):
            return "default"
        for pl in programming_languages:
            if pl.lower() in input_text:
                return "code_to_code_translation"
        return "default"
    elif scenario == "solving_non_stem_exam_question":
        if has_math_expr(input_text):
            return "solving_stem_exam_question"
        return "solving_non_stem_exam_question"
    elif scenario == "recommendation":
        if "recommendation letter" in input_text.lower():
            return "functional_writing"
        cand_list = ["recommendation", "recommend"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "recommendation"
        return "default"
    elif scenario == "language_polishing":
        cand_list = ["polish", "proofread"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "language_polishing"
        return "instructional_rewriting"
    elif scenario == "writing_social_media_post":
        cand_list = ["social media", "tweet", "twitter", "facebook", "instagram", "wechat", "weibo", "linkedin",
                     "reddit"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "writing_social_media_post"
        return "default"
    elif scenario == "title_generation":
        cand_list = ["title", "tittle", "caption"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "title_generation"
        return "default"
    elif scenario == "code_simplification":
        if not has_code_snippet(input_text):
            return "default"
        cand_list = ["optimize", "optimization", "optimizing", "refactor", "compact", "rewrite", "make this code more",
                     "优化"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "code_simplification_refactoring"
        return "default"
    elif scenario == "writing_job_application":
        cand_list = ["cover letter", "求职信", "job application"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "writing_job_application"
        if "write a resume" in input_text.lower():
            return "functional_writing"
        return "default"
    elif scenario == "keywords_extraction":
        cand_list = ["keyword", "keywords", "key word", "key words", "key point", "key points"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "keywords_extraction"
        return "default"
    elif scenario == "writing_news_article":
        cand_list = ["news", "report", "journalism", "journalist", "journal", "新闻", "报道", "记者"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "writing_news_article"
        return "default"
    elif scenario == "writing_biography":
        cand_list = ["biography", "biographies", "传记", " bio "]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "writing_biography"
        return "default"
    elif scenario == "note_summarization":
        cand_list = ["note", "notes", "笔记", "课堂笔记", "课堂记录"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "note_summarization"
        return "default"
    elif scenario == "writing_scientific_paper":
        cand_list = ["paper", "论文", "学术论文", "科研论文", "学术写作", "科研写作"]
        for word in cand_list:
            if word.lower() in input_text.lower():
                return "writing_scientific_paper"
        return "default"
    return scenario


def down_sample_pool(pools, alpha, upper_limit, min_per_class=None):
    # pools is like {"key1": a list of items, "key2": a list of items ... }
    # what you need to do is
    # 1. calculate the fraction of each key over the total number of items
    # 2. get a new fraction by old_fraction^alpha
    # 3. normalize the new fraction so that the sum of all new fractions is 1
    # 4. calculate the number of items to keep for each key: upper_limit*new_fraction
    # 5. randomly sample the items for each key with the number of items to keep
    # 6. if the number of items to keep is larger than the number of items in the key, keep all items

    # return a new pools with the same structure as the input pools
    # and the number of items in the new pools is upper_limit

    # step 1
    total_num = sum([len(v) for v in pools.values()])
    assert total_num >= upper_limit
    fractions = {k: len(v) / total_num for k, v in pools.items()}
    # step 2
    new_fractions = {k: v ** alpha for k, v in fractions.items()}
    # step 3
    new_fractions = {k: v / sum(new_fractions.values()) for k, v in new_fractions.items()}
    # step 4, if the value is a float, we use a random number to decide whether to keep it
    num_to_keep_raw = {k: upper_limit * v for k, v in new_fractions.items()}
    num_to_keep = {}
    for k, v in num_to_keep_raw.items():
        if isinstance(v, float):
            if random.random() < v - int(v):
                num_to_keep[k] = int(v) + 1
            else:
                num_to_keep[k] = int(v)
        else:
            num_to_keep[k] = v
    # check if sum(num_to_keep.values()) == upper_limit
    # if not, add 1 for each until the sum is upper_limit
    for k in num_to_keep.keys():
        num_to_keep[k] = min(num_to_keep[k], len(pools[k]))

    while sum(num_to_keep.values()) < upper_limit:
        for k in num_to_keep.keys():
            num_to_keep[k] += 1
            num_to_keep[k] = min(num_to_keep[k], len(pools[k]))
            if min_per_class is not None:
                num_to_keep[k] = max(num_to_keep[k], min_per_class)
            if sum(num_to_keep.values()) == upper_limit:
                break
    # or remove 1 for each until the sum is upper_limit
    while sum(num_to_keep.values()) > upper_limit:
        for k in num_to_keep.keys():
            num_to_keep[k] -= 1
            num_to_keep[k] = max(num_to_keep[k], 0)
            if min_per_class is not None:
                num_to_keep[k] = max(num_to_keep[k], min_per_class)
            if sum(num_to_keep.values()) == upper_limit:
                break

    # step 5
    new_pools = {}
    not_used_pools = {}
    for k, v in num_to_keep.items():
        selected_ids = random.sample(range(len(pools[k])), v)
        new_pools[k] = [pools[k][i] for i in selected_ids]
        not_used_pools[k] = [item for i, item in enumerate(pools[k]) if i not in selected_ids]
    assert sum([len(v) for v in
                new_pools.values()]) == upper_limit, f"sum([len(v) for v in new_pools.values()])={sum([len(v) for v in new_pools.values()])}, upper_limit={upper_limit}"
    # assert the sum of new_pools and not_used_pools is the same as the original pools
    assert sum([len(v) for v in new_pools.values()]) + sum([len(v) for v in not_used_pools.values()]) == total_num
    return new_pools, not_used_pools


def clean_pairwise(raw_output, protocol, remove_both_neither=True):
    # step 1 check if the format is like
    '''
    1. xxx

        - xxx: xxx

        - xxx: xxx

        ...

    2. xxx

        - xxx: xxx

        - xxx: xxx

        ...
    3. So the final decision is xxx
    '''

    # print(raw_output)
    # raise ValueError

    remove_both_neither_flag = False

    if "So, the final decision is" not in raw_output:
        return raw_output, None, False

    segs = raw_output.split("\n")
    new_segs = []

    if protocol in ['pairwise_notie', 'pairwise_tie']:
        for seg in segs:
            # find how many prefix blank
            # find the position of the first non-blank in this seg

            if seg.strip() == "":
                continue
            else:
                # find the position of the first non-blank in this seg
                if seg.startswith("1. ") and seg.endswith(":"):
                    new_segs.append("1. The criteria that can significantly distinguish the two responses are:")
                elif seg.startswith("2. ") and seg.endswith(":"):
                    new_segs.append(
                        "2. Other important factors that can significantly distinguish the two responses are:")
                elif seg.startswith("3. "):
                    new_segs.append(seg)
                elif any([seg.startswith(str(i) + '.') for i in range(1, 4)]):
                    new_segs.append(seg)
                else:
                    if not seg.startswith('\t'):
                        seg = '\t' + seg.strip()
                    for ii, ch in enumerate(seg):
                        if ch != ' ':
                            break
                    rest_seg = seg[ii:]
                    # check if it looks like `- xxx: xxx`
                    pos = rest_seg.find(': ')
                    if pos == -1:
                        new_segs.append(seg)
                    else:
                        if remove_both_neither:
                            before = rest_seg[:pos]
                            after = rest_seg[pos + 2:]
                            # if "both" or "neither" in after, but no "but" or "however" in after, we abort this line
                            if ("both" in after.lower() or "neither" in after.lower()) and (
                                    "but" not in after.lower() and "however" not in after.lower()):
                                remove_both_neither_flag = True
                                continue
                        new_segs.append(seg)

        # if new_segs have consecutive blank lines, we merge them
        newnew_segs = []
        for seg in new_segs[:-1]:
            newnew_segs.append(seg)
            newnew_segs.append("")
        newnew_segs.append(new_segs[-1])
        final_output = '\n'.join(newnew_segs)

        # print(raw_output)
        # print_colored_text(">>>>>>>>","red")

        assert "So, the final decision is" in final_output
        final_output = final_output.replace('So, the final decision is tied.', 'So, the final decision is Tie.')
        final_output = final_output.replace('So, the final decision is a tie.', 'So, the final decision is Tie.')
        final_output = final_output.replace('So, the final decision is a Tie.', 'So, the final decision is Tie.')

        if "3. So, the final decision is" not in final_output:
            # find "3."
            pos = final_output.rfind("3. ")
            if pos == -1:
                final_output = final_output.replace('So, the final decision is', '3. So, the final decision is')
            else:
                rest = final_output[pos + len("3. "):].strip()
                last_rating_pos = rest.rfind('So, the final decision is')
                final_output = final_output[:pos + len("3. ")] + rest[last_rating_pos:]
    else:
        for seg in segs:
            # find how many prefix blank
            # find the position of the first non-blank in this seg

            if seg.strip() == "":
                continue
            else:
                # find the position of the first non-blank in this seg
                if seg.startswith("1. "):
                    new_segs.append(seg)
                elif seg.startswith("2. ") and seg.endswith(":") and "2. Calculate the volume of the steel plate:" not in seg:
                    new_segs.append("2. The criteria that can significantly distinguish the two responses are:")
                elif seg.startswith("3. ") and seg.endswith(":") and "3. Multiply the volume by the density of steel to get the weight in kilograms:" not in seg:
                    new_segs.append(
                        "3. Other important factors that can significantly distinguish the two responses are:")
                elif seg.startswith("4. "):
                    new_segs.append(seg)
                elif any([seg.startswith(str(i) + '.') for i in range(1, 5)]):
                    new_segs.append(seg)
                else:
                    if not seg.startswith('\t'):
                        seg = '\t' + seg.strip()
                    for ii, ch in enumerate(seg):
                        if ch != ' ':
                            break
                    rest_seg = seg[ii:]
                    # check if it looks like `- xxx: xxx`
                    pos = rest_seg.find(': ')
                    if pos == -1:
                        new_segs.append(seg)
                    else:
                        if remove_both_neither:
                            before = rest_seg[:pos]
                            after = rest_seg[pos + 2:]
                            # if "both" or "neither" in after, but no "but" or "however" in after, we abort this line
                            if ("both" in after.lower() or "neither" in after.lower()) and (
                                    "but" not in after.lower() and "however" not in after.lower()):
                                remove_both_neither_flag = True
                                continue
                        new_segs.append(seg)
        # if new_segs have consecutive blank lines, we merge them
        newnew_segs = []
        for seg in new_segs[:-1]:
            newnew_segs.append(seg)
            newnew_segs.append("")
        newnew_segs.append(new_segs[-1])
        final_output = '\n'.join(newnew_segs)

        assert "So, the final decision is" in final_output
        final_output = final_output.replace('So, the final decision is tied.', 'So, the final decision is Tie.')
        final_output = final_output.replace('So, the final decision is a tie.', 'So, the final decision is Tie.')
        final_output = final_output.replace('So, the final decision is a Tie.', 'So, the final decision is Tie.')

        if "4. So, the final decision is" not in final_output:
            # find "3."
            pos = final_output.find("4. ")
            if pos == -1:
                final_output = final_output.replace('So, the final decision is', '4. So, the final decision is')
            else:
                rest = final_output[pos + len("4. "):].strip()

                last_rating_pos = rest.rfind('So, the final decision is')
                final_output = final_output[:pos + len("4. ")] + rest[last_rating_pos:]

    # check the thing after "So, the final decision is" startswith "Response 1" or "Response 2" or "Tie"
    # if not, we abort this line
    pos = final_output.rfind('So, the final decision is')
    rest = final_output[pos + len('So, the final decision is'):].strip()
    ck_to_label = {'Response 1': 0, 'Response 2': 1, 'Tie': 2, 'neither':2, 'both':2, "that both":2, "that neither":2}
    decision = None

    for item, label in ck_to_label.items():
        if rest.startswith(item):
            decision = label
            break

    return final_output, decision, remove_both_neither_flag


def extract_chosen_criteria(sysmsg):
    chosen_criteria = []
    assert '[Criteria start]' in sysmsg and '[Criteria end]' in sysmsg
    # pick content in between
    criteria = sysmsg[sysmsg.find('[Criteria start]') + len('[Criteria start]'):sysmsg.find('[Criteria end]')].strip()
    lines = criteria.split('\n')
    for line in lines:
        # find line with format like \t10. xxx: yyy
        if line.startswith('\t'):
            criterion_name = line[line.find('.') + 1:line.find(':')].strip()
            chosen_criteria.append(criterion_name)
    return chosen_criteria


def check_criteria_position(cleaned_output, system_msg, protocol):
    chosen_criteria = extract_chosen_criteria(system_msg)
    print(chosen_criteria)
    print(cleaned_output)
    print_colored_text("------------------", "blue")
    return cleaned_output, True


def clean_likert(raw_output, protocol):
    # print(sysmsg)
    # chosen_criteria = extract_chosen_criteria(sysmsg)
    # print(chosen_criteria)
    # print('--------------')
    # print(raw_output)
    # print('--------------')
    # raise ValueError
    segs = raw_output.split("\n")
    new_segs = []
    current_seg = 0

    # check if "So, the final rating is xx/10." is in the output
    if "So, the final rating is" not in raw_output:
        return raw_output, None

    if protocol == 'likert':
        for seg in segs:
            if seg.strip() == "":
                continue
            elif seg.startswith("1. "):
                if seg.endswith(":") or seg.strip() == "1. Review the response and the given criteria to identify **only** the criterion(s) that the response fails to meet very well.":
                    new_segs.append("1. The response fails to meet the following criteria very well:")
                else:
                    new_segs.append(seg)
                current_seg = 1
            elif seg.strip()=="The response meets most of the criteria very well. However, it falls short on the following criterion:":
                continue
            elif seg.startswith("2. "):
                if seg.endswith(
                        ":") or seg.strip() == "2. For each criterion identified in step 1, provide **actionable** suggestions on how the response can be improved.":
                    new_segs.append("2. Suggestions for improvement:")
                else:
                    new_segs.append(seg)
                current_seg = 2
            elif seg.startswith("3. "):
                if seg.endswith(
                        ":") or seg.strip() == "3. Besides the given criteria, brainstorm and provide other suggestions to help improve the response based on your own knowledge.":
                    new_segs.append("3. Other suggestions:")
                else:
                    new_segs.append(seg)
                current_seg = 3
            elif seg.startswith("4. "):
                new_segs.append(seg)
                current_seg = 4
            elif seg.startswith("So, the final rating is") and current_seg == 3:
                new_segs.append(f"4. {seg}")
                current_seg = 4
            elif any([seg.startswith(str(i) + '.') for i in range(1, 5)]):
                new_segs.append(seg)
            else:
                if seg.strip().startswith("The response fails to meet the following criteria very well:"):
                    continue
                if seg.strip().startswith("The response fails to meet the following criteria:"):
                    continue
                if not seg.startswith('\t') and not any([seg.startswith(str(i) + '.') for i in range(1, 6)]):
                    seg = '\t' + seg.strip()
                new_segs.append(seg)

        # make sure a blank line between each segment in new_segs
        newnew_segs = []
        for seg in new_segs[:-1]:
            newnew_segs.append(seg)
            newnew_segs.append("")
        newnew_segs.append(new_segs[-1])
        final_output = '\n'.join(newnew_segs)

        if "4. So, the final rating is" not in final_output:
            # find "4."
            pos = final_output.rfind("4. ")
            if pos == -1:
                return raw_output, None
            # print(final_output)
            # print('--------------')
            rest = final_output[pos + len("4. "):].strip()
            last_rating_pos = rest.rfind('So, the final rating is')
            # put last_rating_pos directly after 4.
            final_output = final_output[:pos + len("4. ")] + rest[last_rating_pos:]

            # print(final_output)
            # print("=================")

        # if so, we extract the xx
        pos = raw_output.rfind('So, the final rating is')
        rest = raw_output[pos + len('So, the final rating is'):].strip()

        pos_a = rest.find('/')
        rating = rest[:pos_a].strip()
        restrest = rest[pos_a + 1:].strip()
        assert restrest.startswith('10')
        # or with a decimal point
        # print(rating)
        assert rating.isdigit() or (rating.replace('.', '').isdigit() and rating.count('.') == 1)
    else:
        for seg in segs:
            if seg.strip() == "":
                continue
            elif seg.startswith("1. "):
                new_segs.append(seg)
                current_seg = 1
            elif seg.startswith("2. "):
                if seg.endswith(":"):
                    new_segs.append("2. The response fails to meet the following criteria very well:")
                else:
                    new_segs.append(seg)
                current_seg = 2
            elif seg.startswith("3. "):
                if seg.endswith(":"):
                    new_segs.append("3. Suggestions for improvement:")
                else:
                    new_segs.append(seg)
                current_seg = 3
            elif seg.startswith("4. "):
                if seg.endswith(":"):
                    new_segs.append("4. Other suggestions:")
                else:
                    new_segs.append(seg)
                current_seg = 4
            elif seg.startswith("5. "):
                new_segs.append(seg)
                current_seg = 5
            elif seg.startswith("So, the final rating is") and current_seg == 4:
                new_segs.append(f"5. {seg}")
                current_seg = 5
            elif any([seg.startswith(str(i) + '.') for i in range(1, 6)]):
                new_segs.append(seg)
            else:
                if seg.strip().startswith("The response fails to meet the following criteria very well:"):
                    continue
                if seg.strip().startswith("The response fails to meet the following criteria:"):
                    continue
                if not seg.startswith('\t') and not any([seg.startswith(str(i) + '.') for i in range(1, 6)]):
                    seg = '\t' + seg.strip()
                new_segs.append(seg)

        # make sure a blank line between each segment in new_segs
        newnew_segs = []
        for seg in new_segs[:-1]:
            newnew_segs.append(seg)
            newnew_segs.append("")
        newnew_segs.append(new_segs[-1])
        final_output = '\n'.join(newnew_segs)

        if "5. So, the final rating is" not in final_output:
            # find "4."
            pos = final_output.rfind("5. ")
            if pos == -1:
                return raw_output, None
            # print(final_output)
            # print('--------------')
            rest = final_output[pos + len("5. "):].strip()
            last_rating_pos = rest.rfind('So, the final rating is')
            # put last_rating_pos directly after 5.
            final_output = final_output[:pos + len("5. ")] + rest[last_rating_pos:]

            # print(final_output)
            # print("=================")

        # if so, we extract the xx
        pos = raw_output.rfind('So, the final rating is')
        rest = raw_output[pos + len('So, the final rating is'):].strip()

        pos_a = rest.find('/')
        rating = rest[:pos_a].strip()
        restrest = rest[pos_a + 1:].strip()
        assert restrest.startswith('10')
        # or with a decimal point
        # print(rating)
        assert rating.isdigit() or (rating.replace('.', '').isdigit() and rating.count('.') == 1)

    return final_output, rating


def check_judgement_format(judgement, protocol):
    # should be like
    prev_seg_id = 0
    segs = judgement.split('\n')

    checks = [False, False, False, False]

    for seg in segs:
        if any([seg.startswith(str(i) + '.') for i in range(1, 6)]):
            if seg.startswith('1. The criteria that can significantly distinguish the two responses are:'):
                checks[1] = True
            if seg.startswith("2. Other important factors that can significantly distinguish the two responses are:"):
                checks[2] = True
            if seg.startswith("2. The criteria that can significantly distinguish the two responses are:"):
                checks[2] = True
            if seg.startswith("3. Other important factors that can significantly distinguish the two responses are:"):
                checks[3] = True
            # line of index-starting
            curr_seg_id = int(seg[0])
            if curr_seg_id != prev_seg_id + 1:
                return False, f"Segment skip error: curr_seg_id={curr_seg_id}, prev_seg_id={prev_seg_id}"
            prev_seg_id = curr_seg_id
        elif seg.strip() == "":
            continue
        else:
            if not seg.startswith('\t'):
                return False, f"Segment without tab error: {seg}"

    # if check['2'] = True, then at least we should find a seg in segs that with current_seg_id=2 and not startswith('2. ') and non empty
    curr_seg_id = 0
    for i in range(1, 4):
        if checks[i]:
            found = False
            for seg in segs:
                if any([seg.startswith(str(i) + '.') for i in range(1, 6)]):
                    curr_seg_id = int(seg[0])
                elif seg.strip() == "":
                    continue
                else:
                    if curr_seg_id == i:
                        found = True
                        break
            if not found:
                return False, f"Segment not found error: {i} \n {judgement}"

    return True, None


def exchange_response1and2(judgement):
    # replace all "Response 1" with "Response 2" and vice versa

    # replace "Response 1" with "Response 3"
    judgement = judgement.replace('Response 1', 'Response 3')
    # replace "Response 2" with "Response 1"
    judgement = judgement.replace('Response 2', 'Response 1')
    # replace "Response 3" with "Response 2"
    judgement = judgement.replace('Response 3', 'Response 2')

    return judgement


def insert_sth_between_each_compontent(alist, nd_insert):
    # insert nd_insert between each component in alist
    new_list = []
    for i in range(len(alist) - 1):
        new_list.append(alist[i])
        new_list.append(nd_insert)
    new_list.append(alist[-1])
    return new_list


def remove_non_alphabetical_characters_head_and_tail(s):
    first_alpha_index = -1
    last_alpha_index = -1
    for i in range(len(s)):
        if s[i].isalpha():
            first_alpha_index = i
            break
    for i in range(len(s) - 1, -1, -1):
        if s[i].isalpha():
            last_alpha_index = i
            break
    return s[first_alpha_index:last_alpha_index + 1]


def remove_non_alphabetical_characters_head(s):
    first_alpha_index = -1
    for i in range(len(s)):
        if s[i].isalpha():
            first_alpha_index = i
            break
    return s[first_alpha_index:]


def resolve_a_line(line):
    # usually the line looks like this: `- xx: yyy`

    # step we find the left-most `:` and split the line into two parts
    if ":" not in line:
        # we fail to find this pattern, ignore it
        return "none"
    criterion_name = line.split(":")[0].strip()
    criterion_desc = line.split(":")[1].strip()

    # remove non-alphabetical characters on the left and right for criterion_name
    # keep the middle ones

    uncased_pure_criterion_name = remove_non_alphabetical_characters_head_and_tail(criterion_name).lower()

    # print(f"pruified criterion_name: {uncased_pure_criterion_name}")

    return uncased_pure_criterion_name, criterion_desc


def analyze_pairwise_judgement(raw_judgement):
    lines = raw_judgement.split("\n")
    seg_id = 0  # to record which segment is the currecnt line

    lines_by_criteria = []

    lines_by_brainstorming = []

    for line in lines:
        if line.strip() == "":
            continue
        index_line = False
        if line.startswith("1. "):
            seg_id = 1
            index_line = True
        elif line.startswith("2. "):
            seg_id = 2
            index_line = True
        elif line.startswith("3. "):
            seg_id = 3
            index_line = True
        else:
            # they are lines of specific evaluation, usually in the form of `- [criterion name]: xxx` or `- [brainstorming name]: xxx`
            if seg_id == 1 and not index_line:
                # here is the judgement based on our criteria
                lines_by_criteria.append(line)
            elif seg_id == 2 and not index_line:
                lines_by_brainstorming.append(line)

    significant_criteria = []
    significant_other_factors = []

    for line in lines_by_criteria:
        criterion_name, _ = resolve_a_line(line)
        if criterion_name == "none":
            continue
        significant_criteria.append(criterion_name)
    for line in lines_by_brainstorming:
        criterion_name, _ = resolve_a_line(line)
        if criterion_name == "none":
            continue
        significant_other_factors.append(criterion_name)

    return significant_criteria, significant_other_factors


def randomize_order_of_criteria_for_pairwise(raw_judgement):
    index_lines = {}
    index_segs = {}

    lines = raw_judgement.split("\n")
    seg_id = 0  # to record which segment is the currecnt line

    for line in lines:
        if line.strip() == "": continue
        index_line = False
        for i in range(1, 4):
            if line.startswith(f"{i}. "):
                seg_id = i
                assert index_segs.get(i, None) is None  # make sure we don't have duplicate index
                index_lines[i] = line
                index_line = True
                break
        if index_line: continue
        # the following are the seg lines
        assert seg_id != 0
        if seg_id not in index_segs:
            index_segs[seg_id] = []
        line_name, _ = resolve_a_line(line)
        index_segs[seg_id].append({"line_name": line_name, "line_text": line})

    # randomize the order of the criteria

    for k, v in index_segs.items():
        random.shuffle(v)

    # and we need to re-construct the lines

    new_lines = []
    for i in range(1, 4):
        if index_lines.get(i, None) is not None:
            new_lines.append(index_lines[i])
        for item in index_segs.get(i, []):
            new_lines.append(item["line_text"])

    newnewlines = insert_sth_between_each_compontent(new_lines, "")

    return "\n".join(newnewlines)


def save_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def batched_generate_with_write(engine, final_inputs,  output_file_name, batch_size=-1, already_have=0,final_metas=None):
    total_cost=0.0
    if final_metas is not None:
        assert len(final_inputs) == len(final_metas)
    if batch_size != -1:
        print_colored_text(f"[INFO] Batched generation with batch size {batch_size}.", "green")
        for batch_start in range(already_have, len(final_inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(final_inputs))
            batch = final_inputs[batch_start:batch_end]
            outputs = engine.generate_batch(batch)
            # meta is a list of dict, we put the keys into the output
            if final_metas is not None:
                batch_meta = final_metas[batch_start:batch_end]
                for i in range(len(outputs)):
                    outputs[i]['meta'] = batch_meta[i]
            safe_write_jsonl(outputs, output_file_name, mode='a')
            if 'cost' in outputs[0]:
                total_cost+=sum([x['cost'] for x in outputs])
            print_colored_text(f"[INFO] Batch {batch_start}-{batch_end}/{len(final_inputs)} are finished and written. | Accumulated total cost: {total_cost}",
                               "green")
    else:
        print_colored_text(f"[INFO] Full generation {len(final_inputs)} samples at one throughput.", "green")
        outputs = engine.generate_batch(final_inputs)
        # meta is a list of dict, we put the keys into the output
        if final_metas is not None:
            for i in range(len(outputs)):
                outputs[i]['meta'] = final_metas[i]
        safe_write_jsonl(outputs, output_file_name, mode='a')
        if "cost" in outputs[0]:
            total_cost = sum([x['cost'] for x in outputs])
        print_colored_text(f"[INFO] All are finished and written.", "green")
        print_colored_text(f"[INFO] Accumulated total cost: {total_cost}", "green")


def get_engine(engine_type, args):
    from .vllm_utils import VllmEngine
    from .api_utils import OpenAIChat
    assert engine_type in ['vllm', 'openai']
    if args.engine_type == "vllm":
        engine = VllmEngine(args)
    else:
        engine = OpenAIChat(api_key=os.environ.get("OPENAI_API_KEY", None), model=args.model_dir,
                            temperature=args.temperature, max_tokens=args.max_new_tokens, top_p=args.top_p,
                            frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty,
                            request_timeout=120)
    return engine





def pairwise_cleaned_output_to_output(cleaned_output, protocol):
    # print(cleaned_output)
    # print_colored_text(">>>>>>>>","red")

    cleaned_lines = [x.strip() for x in cleaned_output.split("\n") if x.strip() != ""]

    reference_lines = []
    factor_lines = []
    decision_lines = []





    if "reasoning" not in protocol:
        # we should expect something like this
        '''
        1. The criteria that can significantly distinguish the two responses are:
            - xxx: xxx
            - xxx: xxx
        2. Other important factors that can significantly distinguish the two responses are:
            - xxx: xxx
            - xxx: xxx
        3. So, the final decision is xxx
        '''
        # case 1:
        segid = 0
        for line in cleaned_lines:
            if line in ["1. The criteria that can significantly distinguish the two responses are:"]:
                segid = 1
            elif line.startswith("1. The criterion that can significantly distinguish the two responses is "):
                segid = 1
                rest = line[len("1. The criterion that can significantly distinguish the two responses is "):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "1. The criterion that can significantly distinguish the two responses is the first specific criterion: "):
                segid = 1
                rest = line[
                       len("1. The criterion that can significantly distinguish the two responses is the first specific criterion: "):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith("1. The only criterion that can significantly distinguish the two responses is the \""):
                segid = 1
                rest = line[len("1. The only criterion that can significantly distinguish the two responses is the \""):]
                # find the first "
                pos = rest.find(".")
                pos1 = rest.find("\"")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos1])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "1. The most significant criterion that distinguishes the two responses is the \""):
                segid = 1
                rest = line[
                       len("1. The most significant criterion that distinguishes the two responses is the \""):]
                # find the first "
                pos = rest.find(".")
                pos1 = rest.find("\"")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos1])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "1. The most significant criterion that distinguishes the two responses is the '"):
                segid = 1
                rest = line[
                       len("1. The most significant criterion that distinguishes the two responses is the '"):]
                # find the first "
                pos = rest.find(".")
                pos1 = rest.find("'")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos1])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith("1. The significant criterion that distinguishes the two responses is"):
                segid = 1
                rest = line[len("1. The significant criterion that distinguishes the two responses is"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith("1. The most significant criterion that distinguishes the two responses is the first one under the specific criteria:"):
                segid = 1
                rest = line[len("1. The most significant criterion that distinguishes the two responses is the first one under the specific criteria:"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith("1. The most significant criterion that distinguishes the two responses is the first content-specific criterion:"):
                segid = 1
                rest = line[len("1. The most significant criterion that distinguishes the two responses is the first content-specific criterion:"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "1. The criterion that can significantly distinguish the two responses is the first specific criterion: "):
                segid = 1
                rest = line[
                       len("1. The criterion that can significantly distinguish the two responses is the first specific criterion: "):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "1. The most significant criterion that distinguishes the two responses is the first specific criterion:"):
                segid = 1
                rest = line[
                       len("1. The most significant criterion that distinguishes the two responses is the first specific criterion:"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith("1. The criteria that can significantly distinguish the two responses are"):
                segid = 1
            elif line.startswith("1. The criteria that significantly distinguish the two responses are"):
                segid = 1
            elif line.startswith("2. Other important factors that can significantly distinguish the two responses are:"):
                segid = 2
            elif line.startswith("2. Other important factors:"):
                segid = 2
            elif line.startswith("2. Another important factor that can distinguish the two responses is"):
                segid = 2
                # find the first .
                rest_line = line[len("2. Another important factor that can distinguish the two responses is "):]
                pos = rest_line.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest_line[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest_line[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith("2. Another important factor that can significantly distinguish the two responses is"):
                segid = 2
                # find the first .
                rest_line = line[
                            len("2. Another important factor that can significantly distinguish the two responses is"):]
                pos = rest_line.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest_line[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest_line[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Besides the given criteria, another important factor that can significantly distinguish the two responses is"):
                segid = 2
                # find the first .
                rest_line = line[
                            len("2. Besides the given criteria, another important factor that can significantly distinguish the two responses is"):]
                pos = rest_line.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest_line[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest_line[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Other important factors that can significantly distinguish the two responses include"):
                segid = 2
                rest = line[
                       len("2. Other important factors that can significantly distinguish the two responses include"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Besides the given criteria, an important factor that distinguishes the two responses is the "):
                segid = 2
                rest = line[
                       len("2. Besides the given criteria, an important factor that distinguishes the two responses is the "):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. An important factor that distinguishes the two responses, beyond the given criteria, is the"):
                segid = 2
                rest = line[
                       len("2. An important factor that distinguishes the two responses, beyond the given criteria, is the"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Other important factors that can significantly distinguish the two responses are"):
                segid = 2
                rest = line[
                       len("2. Other important factors that can significantly distinguish the two responses are"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Other important factors that can distinguish the two responses include"):
                segid = 2
                rest = line[
                       len("2. Other important factors that can distinguish the two responses include"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")

            elif line.startswith(
                    "2. Other important factors that distinguish the two responses include"):
                segid = 2
                rest = line[
                       len("2. Other important factors that distinguish the two responses include"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Besides the given criteria, an important factor that can significantly distinguish the two responses is"):
                segid = 2
                rest = line[
                       len("2. Besides the given criteria, an important factor that can significantly distinguish the two responses is"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Besides the given criteria, another important factor that distinguishes the two responses is"):
                segid = 2
                rest = line[
                       len("2. Besides the given criteria, another important factor that distinguishes the two responses is"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Other important factors that can distinguish the two responses are"):
                segid = 2
                rest = line[
                       len("2. Other important factors that can distinguish the two responses are"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Another important factor that distinguishes the two responses is"):
                segid = 2
                rest = line[
                       len("2. Another important factor that distinguishes the two responses is"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. An important factor that can distinguish the two responses is the"):
                segid = 2
                rest = line[
                       len("2. An important factor that can distinguish the two responses is the"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Besides the given criteria, another important factor that can distinguish the two responses is "):
                segid = 2
                rest = line[
                       len("2. Besides the given criteria, another important factor that can distinguish the two responses is "):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Besides the given criteria, the important factor that can significantly distinguish the two responses is"):
                segid = 2
                rest = line[
                       len("2. Besides the given criteria, the important factor that can significantly distinguish the two responses is"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Besides the given criteria, the tone of the responses"):
                segid = 2
                rest = line[
                       len("2. Besides the given criteria, the tone of the responses"):]
                # find the first "
                pos = rest.find(".")
                factor_name = "tone"
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. An important factor that can significantly distinguish the two responses is"):
                segid = 2
                rest = line[
                       len("2. An important factor that can significantly distinguish the two responses is"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. An important factor that can significantly distinguish the two responses, especially for this user's query, is"):
                segid = 2
                rest = line[
                       len("2. An important factor that can significantly distinguish the two responses, especially for this user's query, is"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith("3. So, the final decision is"):
                segid = 3
                decision_lines.append(line[len("3. "):])
            elif segid in [1, 2]:
                factor_lines.append(line)
            elif segid == 3:
                decision_lines.append(line)
            else:
                return "bad"

        factor_lines = [re.sub(r'\(Criterion \d+\)', '', x) for x in factor_lines if not x.startswith('[Response')]
        # add \t for all lines
        for i in range(len(factor_lines)):
            if not any([factor_lines[i].startswith(str(j) + '.') for j in range(1, 5)]):
                factor_lines[i] = '\t' + factor_lines[i].strip()
        for i in range(len(decision_lines)):
            if not any([decision_lines[i].startswith(str(j) + '.') for j in range(1, 5)]):
                decision_lines[i] = '\t' + decision_lines[i].strip()

        combined_str = "1. The key factors to distinguish these two responses:\n\n" + "\n\n".join(
            factor_lines) + "\n\n2. The final decision:\n\n" + "\n\n".join(decision_lines)
    else:
        segid = 0
        for line in cleaned_lines:
            if line.startswith("1. "):
                segid = 1
                reference_lines.append("Independent reference response through step-by-step reasoning:")
                rest = line[len("1. "):]
                removed_starting = ["Reference Response: ","Reference for the user's query: "]
                for rs in removed_starting:
                    if rest.startswith(rs):
                        rest = rest[len(rs):]
                reference_lines.append(rest)

            elif line == "2. The criteria that can significantly distinguish the two responses are:":
                segid = 2
            elif line.startswith("2. The criterion that can significantly distinguish the two responses is "):
                segid = 2
                rest = line[len("2. The criterion that can significantly distinguish the two responses is "):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. The criterion that can significantly distinguish the two responses is the first specific criterion: "):
                segid = 2
                rest = line[
                       len("2. The criterion that can significantly distinguish the two responses is the first specific criterion: "):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Upon reviewing the two responses, the criterion that can significantly distinguish the two responses is"):
                segid = 2
                rest = line[
                       len("2. Upon reviewing the two responses, the criterion that can significantly distinguish the two responses is"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "2. Based on the given criteria, the significant distinguishing factor between the two responses is"):
                segid = 2
                rest = line[
                       len("2. Based on the given criteria, the significant distinguishing factor between the two responses is"):]
                # find the first "
                pos = rest.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "3. Other important factors that can significantly distinguish the two responses are:"):
                segid = 3
            elif line.startswith("3. Other Important Factors:"):
                segid = 3
            elif line.startswith("3. Another important factor that can distinguish the two responses is"):
                segid = 3
                # find the first .
                rest_line = line[len("3. Another important factor that can distinguish the two responses is "):]
                pos = rest_line.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest_line[:pos])
                # capitalize the first letter
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest_line[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith("3. Another important factor that can significantly distinguish the two responses is"):
                segid = 3
                # find the first .
                rest_line = line[
                            len("3. Another important factor that can significantly distinguish the two responses is"):]
                pos = rest_line.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest_line[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest_line[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "3. Besides the given criteria, another important factor that can significantly distinguish the two responses is"):
                segid = 3
                # find the first .
                rest_line = line[
                            len("3. Besides the given criteria, another important factor that can significantly distinguish the two responses is"):]
                pos = rest_line.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest_line[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest_line[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith(
                    "3. Other important factors that can significantly distinguish the two responses include"):
                segid = 3
                # find the first .
                rest_line = line[
                            len("3. Other important factors that can significantly distinguish the two responses include"):]
                pos = rest_line.find(".")
                factor_name = remove_non_alphabetical_characters_head_and_tail(rest_line[:pos])
                factor_name = factor_name[0].upper() + factor_name[1:]
                description = remove_non_alphabetical_characters_head(rest_line[pos + 1:])
                factor_lines.append(f"- {factor_name}: {description}")
            elif line.startswith("4. So, the final decision is"):
                segid = 4
                decision_lines.append(line[len("4. "):])
            elif segid == 1:
                reference_lines.append(line)
            elif segid in [2, 3]:
                factor_lines.append(line)
            elif segid == 4:
                decision_lines.append(line)
            else:
                return "bad"

        factor_lines = [re.sub(r'\(Criterion \d+\)', '', x) for x in factor_lines if not x.startswith('[Response')]

        for i in range(len(reference_lines)):
            if not any([reference_lines[i].startswith(str(j) + '.') for j in range(1, 5)]):
                reference_lines[i] = '\t' + reference_lines[i].strip()
        for i in range(len(factor_lines)):
            if not any([factor_lines[i].startswith(str(j) + '.') for j in range(1, 5)]):
                factor_lines[i] = '\t' + factor_lines[i].strip()
        for i in range(len(decision_lines)):
            if not any([decision_lines[i].startswith(str(j) + '.') for j in range(1, 5)]):
                decision_lines[i] = '\t' + decision_lines[i].strip()

        combined_str = "\n\n".join(
            reference_lines) + "\n\n2. The key factors to distinguish these two responses:\n\n" + "\n\n".join(
            factor_lines) + "\n\n3. The final decision:\n\n" + "\n\n".join(decision_lines)
        # print(combined_str)
        # print_colored_text(">>>>>>>>", "blue"


    return reference_lines,factor_lines,decision_lines



def keep_head_and_tail(input_string, length,tokenizer):
    raw_tokens = tokenizer(input_string, truncation=False, add_special_tokens=False, max_length=100000)['input_ids']
    if len(raw_tokens) < length:
        # short enough to input all
        return input_string
    else:
        # take the first length/2 tokens and last length/2 tokens, and add ... in between
        return tokenizer.decode(raw_tokens[:length // 2]) + " ... " + tokenizer.decode(
            raw_tokens[-length // 2:])



def grammar_check(sstr):
    """
    Version 0.1: given a string, return the grammar errors, determined by `language_tool_python`
    we first split the str into sentences and then check each sentence
    :param sstr: a atring
    :return: a list of grammar mistakes
    """
    if grammarcktool is None:
        raise ValueError("grammarcktool is not initialized")
    # sstr = 'A sentence with a error in the Hitchhiker’s Guide tot he Galaxy'
    # tokenize as sentences first

    sents = sent_tokenize(sstr)
    all_grammar_errors = []
    num_all_errors = 0
    for sent in sents:
        inner_sent = grammarcktool.check(sent)
        strlike_errors = [x.__str__() for x in inner_sent]
        all_grammar_errors.extend(strlike_errors)
        num_all_errors += len(strlike_errors)

    return all_grammar_errors,num_all_errors

    # all_grammar_errors = grammarcktool.check(sstr)
    # pprint(all_grammar_errors)


if __name__ == '__main__':
    sstr = """ Sure, here's a minimal decision transformer implemented in PyTorch! Heres an good acse!"""
    res = grammar_check(sstr)
    elegant_show(res,full=True)
