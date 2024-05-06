import openai

from shared.codes.utils import *
from shared.codes.api_utils import *
import argparse
from shared.codes.prompt_utils import *

openai.api_base = "https://lonlie.plus7.plus/v1"
# api_key = "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"
openai_api_key = "sk-YYpAQgm6ntC5RNIA3a85E35238D845Fa807f087dDf61D4Da"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt-3.5-turbo-1106")
    parser.add_argument("--ask_AB_prompt", default="naive")
    parser.add_argument("--change_AB", default=False, action="store_true")
    parser.add_argument("--end_pos", type=int, default=1000000000, )
    parser.add_argument("--batch_size", type=int, default=20, )
    args = parser.parse_args()

    api_engine = OpenAIChat(api_key=openai_api_key, model=args.model_name,
                            temperature=0.0, max_tokens=3, top_p=1.0,
                            frequency_penalty=0, presence_penalty=0, request_timeout=120,
                            type="text", seed=42)

    tokenizer = None
    data = read_all("./shared/data/chatbot_arena_shuffled_no-tie.jsonl")

    flag = "AB" if not args.change_AB else "BA"
    output_file = f"collected_data/model_preference/{args.model_name}/direct_ask_{args.ask_AB_prompt}_preferences_{flag}.jsonl"
    start_pos = len(read_all(output_file))
    all_wrapped_q_rpairs = []
    for item in data[start_pos:args.end_pos]:
        query = item["prompt"]
        r1 = item["response_a"]
        r2 = item["response_b"]
        if args.change_AB:
            r1, r2 = r2, r1
        wrapped_p_rpair = wrapper_p_rpair(wrapper_type=args.ask_AB_prompt, query=query, response_1=r1, response_2=r2,
                                          tokenizer=tokenizer)
        all_wrapped_q_rpairs.append({"usermsg":wrapped_p_rpair})
        # print(wrapped_p_rpair)
        # raise ValueError

    batched_generate_with_write(api_engine, all_wrapped_q_rpairs, output_file, batch_size=args.batch_size)
