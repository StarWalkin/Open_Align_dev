from transformers import AutoTokenizer
def wrapper_io(wrapper_type, query, response, tokenizer):
    if wrapper_type == "llama-2-chat":
        p1 = f"[INST] {query} [/INST]"
        p2 = f" {response} </s>"
    elif wrapper_type == "mistral":
        p1 = f"[INST] {query} [/INST]"
        p2 = f"{response}</s>"
    elif wrapper_type == "vicuna":
        p1 = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {query} ASSISTANT:"
        p2 = f" {response} </s>"
    elif wrapper_type == "tulu":
        p1 = f"<|user|>\n{query}\n<|assistant|>\n"
        p2 = f"{response}</s>"
    elif wrapper_type == "yi":
        p1 = f"<|im_start|> user\n{query}<|im_end|> \n<|im_start|> assistant"
        p2 = f"{response}<|im_end|>"
    elif wrapper_type == "zephyr":
        p1 = f"<|user|>\n{query}</s>\n<|assistant|>\n"
        p2 = f"{response}</s>"
    else:
        raise NotImplementedError
    p1_tokens = tokenizer.encode(p1) # automatically, we have the <s> token
    p1p2_tokens = tokenizer.encode(p1+p2) # automatically, we have the <s> token
    num_p1_tokens = len(p1_tokens)
    p2_tokens = p1p2_tokens[num_p1_tokens:]
    num_p2_tokens = len(p2_tokens)

    return num_p1_tokens,num_p2_tokens,p1+p2

def wrapper_input(wrapper_type, tokenizer, conversation):
    if wrapper_type in ['llama-2-chat','yi','zephyr','qwen']:
        jx = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # if jx.startswith(tokenizer.bos_token):
        #     jx = jx[len(tokenizer.bos_token):]
        return jx
    elif wrapper_type == "alpaca":
        assert len(conversation) in [1,3]
        if len(conversation)==1:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{conversation[0]['content']}\n\n### Response:"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{conversation[0]['content']}\n\n### Response:{conversation[1]['content']}\n\n### Instruction:\n{conversation[2]['content']}"
    else:
        raise NotImplementedError

def wrapper_p_rpair(wrapper_type,query,response_1,response_2):
    sysprompt = None
    if wrapper_type=="naive":
        strs = """You will need to analyze two responses (Response A and Response B) from AI assistants to a user's query. The query and the responses are as follows:

[Query Start]
which is the best city in the world?
[Query End]

[Response A Start]
beijing
[Response A End]

[Response B Start]
seattle
[Response B End]

Between Response A and Response B, which response is better in addressing the user's query? The better response is Response"""
    elif wrapper_type=="naive_openai":
        strs = """You will need to analyze two responses (Response A and Response B) from AI assistants to a user's query. The query and the responses are as follows:

[Query Start]
{prompt}
[Query End]

[Response A Start]
{response_1}
[Response A End]

[Response B Start]
{response_2}
[Response B End]

Between Response A and Response B, which response is better in addressing the user's query? You must start your answer with "A" or "B" without exception."""
    elif wrapper_type=="debug":
        strs = query[:10]
        return strs
    elif wrapper_type=="alpaca_eval":
        raise ValueError("AlpacaEval do not requires specific prompt, we skip it and use naive_openai instead.")
    elif wrapper_type=="mt_bench":
        sysprompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI 
        assistants to the user question displayed below. You should choose the assistant that follows the user’s 
        instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, 
        relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure 
        that the order in which the responses were presented does not influence your decision. Do not allow the length of 
        the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible."""
        strs = """[User Question]
{prompt}
	
[The Start of Assistant A's Answer]
{response_1}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_2}
[The End of Assistant B's Answer]

Between Assistant A and Assistant B, which is better? You must start your answer with "A" or "B" without exception."""
    else:
        raise NotImplementedError

    return sysprompt, strs.format(prompt=query,response_1=response_1,response_2=response_2)


if __name__ == '__main__':
    ckpt_dir = "/cpfs01/shared/GAIR/GAIR_hdd/ckpts"
    tokenizer = AutoTokenizer.from_pretrained(f"{ckpt_dir}/Qwen/Qwen-1_8B-Chat",trust_remote_code=True)

    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
        {"role": "assistant", "content": "A pseudo response!!!"},
        {"role": "user", "content": "Are you serious?"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print('-----------')
    print(prompt)
    print('-----------')
