import json
import random
import re
from pathlib import Path

import tiktoken
from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS


@LOAD_DATASET.register_module()
class CDMEDataset(BaseDataset):

    @staticmethod
    def load(
        path: str,
        length: int,
        depth: int,
        tokenizer_model: str,
        file_list: list[str],
        num_repeats_per_file: int,
        length_buffer: int,
        guide: bool,
        language: str,
        needle: str,
        retrieval_question: str,
    ):
        data = {'prompt': [], 'answer': []}
        tokenizer = tiktoken.encoding_for_model(tokenizer_model)

        def _generate_context(tokens_context, depth_percent, needle):
            tokens_needle = _get_tokens_from_context(needle)
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_context = (tokens_context[:insertion_point] +
                              tokens_needle + tokens_context[insertion_point:])
            new_context = _decode_tokens(tokens_context)
            return new_context

        def _get_tokens_from_context(context):
            return tokenizer.encode(context)

        def _decode_tokens(tokens):
            return tokenizer.decode(tokens)

        def _modify_retrieval_question(retrieval_question):
            if language == 'Chinese':
                parts = retrieval_question.split('请按照')
                guide_retrieval_question = (parts[0] + '在回答之前，请思考文档中与此问题'
                                            '最相关的内容是什么。请按照' + parts[1])
                return guide_retrieval_question
            elif language == 'English':
                parts = retrieval_question.split('Please answer in the format')
                guide_retrieval_question = (
                    parts[0] + 'Before answering, please consider'
                    ' what in the document is most relevant to this question.'
                    ' Please answer in the format' + parts[1])
                return guide_retrieval_question
            else:
                raise ValueError(f"Language '{language}' is not supported.")

        def _generate_prompt(context, retrieval_question):
            if guide:
                retrieval_question = _modify_retrieval_question(
                    retrieval_question)

            if language == 'Chinese':
                prompt = ('你是一个善于回答用户问题的智能AI助手\n'
                          '请保持你的回答简洁清楚。不要说和下面文档中的无关的话'
                          '，或重复你的回答\n'
                          f'用户现在给你的文档是{context}\n\n'
                          f'现在请问：{retrieval_question}')
            elif language == 'English':
                prompt = ('You are an intelligent AI assistant skilled in '
                          'answering user questions.\n'
                          'Please keep your answers concise and clear. Do not'
                          ' talk about irrelevant topics or repeat your '
                          'answers.\n'
                          f'The document given to you by the user is {context}'
                          f'\n\nNow, the question is: {retrieval_question}')
            else:
                raise ValueError(f"Language '{language}' is not supported.")

            return prompt

        files = Path(path).glob('*.jsonl')
        for file in files:
            if file.name not in file_list:
                continue

            with open(file, 'r', encoding='utf-8') as f:
                lines_bak = [json.loads(line.strip()) for line in f]
            lines = lines_bak.copy()
            for counter in range(num_repeats_per_file):
                random.seed(counter)
                random.shuffle(lines)

                context_length = length - length_buffer
                target_length_per_record = context_length - len(
                    _get_tokens_from_context(needle))

                accumulated_tokens = []
                for line in lines:
                    tokens_current_line = _get_tokens_from_context(
                        line['text'])
                    accumulated_tokens.extend(tokens_current_line)

                    if len(accumulated_tokens) >= target_length_per_record:
                        break

                processed_text = _generate_context(
                    accumulated_tokens[:target_length_per_record], depth,
                    needle)

                processed_prompt = _generate_prompt(processed_text,
                                                    retrieval_question)

                data['prompt'].append(processed_prompt)
                data['answer'].append(needle)

        dataset = Dataset.from_dict({
            'prompt': data['prompt'],
            'answer': data['answer'],
        })
        return dataset


class CDMEEvaluator(BaseEvaluator):

    def __init__(self, use_trim=False):
        self.use_trim = use_trim

    @staticmethod
    def _trim_prediction(prediction, reference):
        """Trims the prediction string based on the length of the reference
        string.

        Args:
            prediction (str): The prediction string.
            reference (str): The reference string.

        Returns:
            str: The trimmed prediction string.
        """
        l08 = int(0.8 * len(reference))
        l12 = int(1.2 * len(reference))
        trimmed_prediction = prediction[:l12]

        if len(trimmed_prediction) > l08 and \
                reference[-1] in trimmed_prediction[l08:]:
            end_pos = l08 + trimmed_prediction[l08:].index(reference[-1]) + 1
            trimmed_prediction = trimmed_prediction[:end_pos]

        return trimmed_prediction

    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different lengths'
            }

        total_score = 0
        details = []
        for prediction, reference in zip(predictions, references):
            prediction = re.sub(r'\s+', '', prediction)
            reference = re.sub(r'\s+', '', reference)

            if self.use_trim:
                prediction = CDMEEvaluator._trim_prediction(
                    prediction, reference)

            edit_distance = self.levenshtein_distance(prediction, reference)
            max_len = max(len(prediction), len(reference))
            score = 100 * (1 -
                           edit_distance / max_len) if max_len != 0 else 100

            detail = {
                'pred': prediction,
                'answer': reference,
                'edit_distance': edit_distance,
                'score': score
            }
            total_score += score
            details.append(detail)

        average_score = total_score / len(predictions) if predictions else 0
        result = {'score': average_score, 'details': details}
        return result


@TEXT_POSTPROCESSORS.register_module('cdme')
def cdme_postprocess(text: str) -> str:
    return text


@TEXT_POSTPROCESSORS.register_module('cdme_dataset')
def cdme_dataset_postprocess(text: str) -> str:
    return text
