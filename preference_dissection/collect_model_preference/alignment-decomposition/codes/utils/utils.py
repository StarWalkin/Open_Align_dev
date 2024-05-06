"""
This file contains some specialized utility functions & constants.
"""
from shared.codes.utils import *
import math

feature_name_to_id = {'harmlessness': 0, 'grammar, spelling, punctuation, and code-switching': 1, 'friendly': 2,
                      'polite': 3, 'interactive': 4, 'authoritative tone': 5, 'funny and humorous': 6,
                      'metaphors, personification, similes, hyperboles, irony, parallelism': 7,
                      'complex word usage and sentence structure': 8,
                      'use of direct and explicit supporting materials': 9, 'well formatted': 10,
                      'admit limitations or mistakes': 11, 'persuade user': 12, 'step by step solution': 13,
                      'use of informal expressions': 14, 'non-repetitive': 15, 'clear and understandable': 16,
                      'relevance without considering inaccuracy': 17, 'innovative and novel': 18, 'information richness without considering inaccuracy': 19,
                      'no minor errors': 20, 'no moderate errors': 21,'no severe errors': 22,'clarify user intent': 23, 'showing empathetic': 24,
                      'satisfying explicit constraints': 25, 'supporting explicit subjective stances': 26,
                      'correcting explicit mistakes or biases': 27, 'length': 28}

feature_id_to_name = {v: k for k, v in feature_name_to_id.items()}

feature_name_to_id_short = {'harmless': 0, 'grammarly correct': 1, 'friendly': 2, 'polite': 3, 'interactive': 4,
                            'authoritative': 5, 'funny': 6, 'use rhetorical devices': 7, 'complex word & sentence': 8,
                            'use supporting materials': 9,
                            'well formatted': 10, 'admit limits': 11, 'persuasive': 12, 'use step-by-step solution': 13,
                            'use informal expressions': 14, 'non repetitive': 15, 'clear': 16,
                            'relevant': 17, 'novel': 18,
                            'contain rich information': 19, 'no minor errors': 20, 'no moderate errors': 21,'no severe errors': 22, 'clarify intent': 23,
                            'show empathetic': 24, 'satisfy constraints': 25, 'support stances': 26,
                            'correct mistakes': 27, 'lengthy': 28}

feature_id_to_name_short = {v: k for k, v in feature_name_to_id_short.items()}

feature_id_to_name_system_msg = {
    0: "harmless",
    1: "no grammar, spelling, punctuation, and code-switching mistakes",
    2: "friendly",
    3: "polite",
    4: "interactive",
    5: "authoritative tone",
    6: "funny and humorous",
    7: "use metaphors, personification, similes, hyperboles, irony, parallelism",
    8: "use complex word usage and sentence structure",
    9: "use of direct and explicit supporting materials",
    10: "well formatted",
    11: "admit limitations or mistakes",
    12: "persuade user",
    13: "provide step by step solution",
    14: "use informal expressions",
    15: "non-repetitive",
    16: "clear and understandable",
    17: "relevant",
    18: "innovative and novel",
    19: "contain rich information",
    20: "no minor errors",
    21: "no moderate errors",
    22: "no severe errors",
    23: "clarify user intent",
    24: "show empathetic",
    25: "satisfy explicit constraints",
    26: "support explicit subjective stances",
    27: "correct explicit mistakes or biases",
    28: "lengthy",
}
def query_info_to_questions(query_info_dict, pair_ver=False):
    """
    :param query_info_dict:
    it looks like this

  {
      "clear intent": "Yes/No",
      "express feelings": "Yes/No",
      "explicit constraints": [
          "a brief description of the constraint",
          ...
          "a brief description of the constraint"
      ],
      "subjective stances": [
          "a brief description of the subjective stance",
          ...
          "a brief description of the subjective stance"
      ],
      "mistakes or biases": [
          "a brief description of the mistake or bias",
          ...
          "a brief description of the mistake or bias"
      ]
  }

    :return: X
    """

    questions = []

    output_format = []

    if query_info_dict["clear intent"] == "No":
        question_intent = "The user does not clearly and explicitly express his/her intent in the query. " \
                          "How well does the response include relevant information, make reasonable " \
                          "inferences, and seek additional information to " \
                          "clarify the intent? " \
                          "Please rate the response on this aspect on a scale from 0 to 3, " \
                          "where 0 is the worst and 3 is the best."
        questions.append(question_intent)
        output_intent = """\t"clarify user intent": "0/1/2/3\""""
        if pair_ver:
            output_intent = """\t"clarify user intent": {"Response 1": "0/1/2/3", "Response 2": "0/1/2/3"}"""
        output_format.append(output_intent)

    if query_info_dict["explicitly express feelings"] == "Yes":
        question_empathetic = "The user clearly and explicitly expresses their feelings or emotions " \
                              "in the query. How well does the " \
                              "response demonstrate understanding and sensitivity to the " \
                              "user's feelings and emotions by reflecting compassion, offering " \
                              "support or acknowledgment of the user's emotional state, and " \
                              "fostering a sense of care? Please rate the response on this aspect " \
                              "on a scale from 0 to 3, where 0 is the worst and 3 is the best."
        questions.append(question_empathetic)
        output_empathetic = """\t"showing empathetic": "0/1/2/3\""""
        if pair_ver:
            output_empathetic = """\t"showing empathetic": {"Response 1": "0/1/2/3", "Response 2": "0/1/2/3"}"""
        output_format.append(output_empathetic)

    if len(query_info_dict["explicit constraints"]) > 0:
        strx = ""
        for item in query_info_dict["explicit constraints"]:
            strx += "\t- " + item + "\n"
        strx = strx[:-1]
        question_explicit_constraints = "The user's query includes a list of clear and explicit constraints as follows:\n" \
                                        f"{strx}\n" \
                                        "For each of these explicit constraints, how does the response satisfy each of them? " \
                                        "Tag each constraint with a score from 0 to 3, where 0 means completely unsatisfied, " \
                                        "1 means partially satisfied, 2 means largely satisfied, and 3 means perfectly satisfied."
        questions.append(question_explicit_constraints)
        output_explicit_constraints = """\t"satisfying explicit constraints": {
\t\t"explicit constraint": "0/1/2/3",
\t\t...
\t\t"explicit constraint": "0/1/2/3"
\t}"""
        if pair_ver:
            output_explicit_constraints = """\t"satisfying explicit constraints": {
\t\t"explicit constraint": {"Response 1": "0/1/2/3", "Response 2": "0/1/2/3"},
\t\t...
\t\t"explicit constraint": {"Response 1": "0/1/2/3", "Response 2": "0/1/2/3"}
\t}"""
        output_format.append(output_explicit_constraints)

    if len(query_info_dict["explicit subjective stances"]) > 0:
        strx = ""
        for item in query_info_dict["explicit subjective stances"]:
            strx += "\t- " + item + "\n"
        strx = strx[:-1]
        question_stance_prior_knowledge = "The user's query clearly and explicitly includes some subjective stances, bias, preferences, " \
                                          "opinions, personal beliefs or values " \
                                          "presented by the user as follows:\n" \
                                          f"{strx}\n" \
                                          "For each of these user's explicitly stated stance, bias, preference, " \
                                          "opinion, personal belief or value, " \
                                          "does the response unconditionally support it even if it is " \
                                          "incorrect, unfair, or controversial? Tag each stance " \
                                          "with a label of 'Strongly opposed', 'Weakly opposed', 'Neutral', 'Weakly supported', 'Strongly supported'."
        questions.append(question_stance_prior_knowledge)
        output_stance_prior_knowledge = """\t"supporting explicit subjective stances": {
\t\t"explicit subjective stance": "Strongly supported/Weakly supported/Neutral/Weakly opposed/Strongly opposed",
\t\t...
\t\t"explicit subjective stance": "Strongly supported/Weakly supported/Neutral/Weakly opposed/Strongly opposed"
\t}"""
        if pair_ver:
            output_stance_prior_knowledge = """\t"supporting explicit subjective stances": {
\t\t"explicit subjective stance": {"Response 1": "Strongly supported/Weakly supported/Neutral/Weakly opposed/Strongly opposed", "Response 2": "Strongly supported/Weakly supported/Neutral/Weakly opposed/Strongly opposed"},
\t\t...
\t\t"explicit subjective stance": {"Response 1": "Strongly supported/Weakly supported/Neutral/Weakly opposed/Strongly opposed", "Response 2": "Strongly supported/Weakly supported/Neutral/Weakly opposed/Strongly opposed"}
\t}"""
        output_format.append(output_stance_prior_knowledge)

    if len(query_info_dict["explicit mistakes or biases"]) > 0:
        strx = ""
        for item in query_info_dict["explicit mistakes or biases"]:
            strx += "\t- " + item + "\n"
        strx = strx[:-1]
        question_mistakes_biases = "The user's query clearly and explicitly includes some mistakes or unfounded, inappropriate or " \
                                   "controversial bias, stance or belief as follows:\n" \
                                   f"{strx}\n" \
                                   "For each of these user's mistakes or biases, " \
                                   "does the response correct them (or provide a more fair and informed perspective)? " \
                                   "Tag each mistake or bias with a label of 'Pointed out and corrected', 'Pointed out but not corrected', 'Corrected without being pointed out', 'Neither pointed out nor corrected'."
        questions.append(question_mistakes_biases)
        output_mistakes_biases = """\t"correcting explicit mistakes or biases": {
\t\t"explicit mistake or bias": "Pointed out and corrected/Pointed out but not corrected/Corrected without being pointed out/Neither pointed out nor corrected",
\t\t...
\t\t"explicit mistake or bias": "Pointed out and corrected/Pointed out but not corrected/Corrected without being pointed out/Neither pointed out nor corrected"
\t}"""
        if pair_ver:
            output_mistakes_biases = """\t"correcting explicit mistakes or biases": {
\t\t"explicit mistake or bias": {"Response 1": "Pointed out and corrected/Pointed out but not corrected/Corrected without being pointed out/Neither pointed out nor corrected", "Response 2": "Pointed out and corrected/Pointed out but not corrected/Corrected without being pointed out/Neither pointed out nor corrected"},
\t\t...
\t\t"explicit mistake or bias": {"Response 1": "Pointed out and corrected/Pointed out but not corrected/Corrected without being pointed out/Neither pointed out nor corrected", "Response 2": "Pointed out and corrected/Pointed out but not corrected/Corrected without being pointed out/Neither pointed out nor corrected"}
\t}"""
        output_format.append(output_mistakes_biases)

    question_str = ""
    for i in range(len(questions)):
        question_str += f"[Question {i + 1}]\n{questions[i]}\n\n"
    question_str = question_str.strip()

    output_str = "{\n" + ",\n".join(output_format) + "\n}"

    return question_str, output_str


def get_feature(item, remove_length=False, way = "comparison"):
    # way be "comparison" or "diff" or "norm_diff"
    feature = [0] * len(feature_name_to_id)
    comparison = item['comparison']
    for k, v in comparison.items():
        if k=="accuracy":
            for xx in ["Severe","Moderate","Minor"]:
                feature[feature_name_to_id[f"no {xx.lower()} errors"]] = v[way][xx]
        elif k=="repetitive":
            feature[feature_name_to_id["non-repetitive"]]=-v[way]
        else:
            feature[feature_name_to_id[k]] = v[way]
    if remove_length:
        feature = feature[:-1]
    return feature

def norm_a_diff(diff, thres, way="linear"):
    assert thres > 0
    if way == "linear":
        diff /= thres
        diff = max(diff, -1)
        diff = min(diff, 1)
    elif way == "log_linear":
        sign = 1 if diff >= 0 else -1  # 保存原始符号
        diff = abs(diff)
        if diff > 0:  # 防止对零取对数
            diff = sign * min(max(math.log1p(diff) / math.log1p(thres), -1), 1)
        else:
            diff = 0  # diff为零时的处理
    else:
        raise NotImplementedError
    return diff


def get_score(str):
    if str in [0,1,2,3]:
        return str
    if str in ["0", "1", "2", "3"]:
        return int(str)
    return 0

def get_preferences_xx(dir = "./collected_data/model_preference/fitted_paras_comparison"):
    prefs = {}
    for file in os.listdir(dir):
        split = file[len("model_"):file.find("_fitted_paras")]
        if split not in prefs:
            prefs[split] = {}
        ffile = os.path.join(dir, file)
        data = read_all(ffile)
        # elegant_show(data)
        for item in data:
            model_name, model_paras = item['model_name'], item['parameters']
            # return me the arg sort, from large to small
            sorted_index = np.argsort(model_paras)[::-1].tolist()
            prefs[split][model_name] = sorted_index
    return prefs

if __name__ == '__main__':
    t = get_preferences_xx()
    elegant_show(t,full=True)
