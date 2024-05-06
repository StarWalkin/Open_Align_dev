import random

from utils import *
from nltk import word_tokenize, sent_tokenize

def calculate_word(text):
    xx = 0
    for sent in sent_tokenize(text):
        xx += len(word_tokenize(sent))
    return xx

def get_toxic_tag(x):
    os = ""
    if x['roberta-large']['flagged']:
        os+="roberta-large,"
    if x['t5-large']['flagged']:
        os+="t5-large,"
    return os.strip(",")

def get_oaimoderation_tag(x):
    tt = ""
    for k,v in x.items():
        if v:
            tt+=k+","
    return tt.strip(",")

if __name__ == '__main__':
    data = read_all("../data/chatbot_arena_shuffled.jsonl")

    t = []
    idx = 0
    for item in data:
        if item['winner'] in ["model_a",'model_b']:
            item['id']=idx
            idx+=1
            t.append(item)
    write_jsonl(t, "../data/chatbot_arena_shuffled_no-tie.jsonl")
    raise ValueError


    data = read_all("../data/data_en_annotated.jsonl")

    newxx = []

    for item in tqdm(data):
        # elegant_show(item,full=True)
        # raise ValueError

        if item['turn']!=1:continue
        if item['language']!='English':continue
        newsample = {}
        newsample['model_a'] = item['model_a']
        newsample['model_b'] = item['model_b']
        newsample['winner'] = item['winner']
        newsample['scenario'] = item['conversation_a'][0]['scenario' \
                                                          '']
        newsample['prompt'] = item['conversation_a'][0]['content']
        newsample['prompt llama tokens'] = item['conversation_a'][0]['num_llama_tokens']
        newsample['prompt word'] = calculate_word(newsample['prompt'])
        newsample['response_a'] = item['conversation_a'][1]['content']
        newsample['response_b'] = item['conversation_b'][1]['content']
        newsample['response_a llama tokens'] = item['conversation_a'][1]['num_llama_tokens']
        newsample['response_b llama tokens'] = item['conversation_b'][1]['num_llama_tokens']
        newsample['response_a word'] = calculate_word(newsample['response_a'])
        newsample['response_b word'] = calculate_word(newsample['response_b'])
        newsample['oaimoderation_tag'] = get_oaimoderation_tag(item['openai_moderation']['categories'])
        newsample['toxic_chat_tag'] = get_toxic_tag(item['toxic_chat_tag'])

        newxx.append(newsample)

    random.shuffle(newxx)

    write_jsonl(newxx,"../data/chatbot_arena_shuffled_x.jsonl")




