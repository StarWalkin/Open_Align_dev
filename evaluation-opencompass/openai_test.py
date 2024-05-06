from openai import OpenAI
from pprint import pprint
import unicodedata

def msg_wrapper(pred, refer, metric):
    if metric == 'info':
        msg = [{"role":"user","content":'Q: {0}\nA: {1}\nIs it helpful?'.format(refer, pred)}]
        return msg
    elif metric == 'truth':
        msg = [{"role":"user","content":'Q: {0}\nA: {1}\nIs it true?'.format(refer, pred)}]
        return msg
    
pred = 'I am not sure'
refer = '1+1=?'

input_msg = msg_wrapper(pred, refer, 'info')

client = OpenAI(
                    api_key="sk-rg8izdOSsK2tLXLt124865B116214e5bAa98C0Bc0c0b6aBc",
                    base_url="https://lonlie.plus7.plus/v1"
                )

completion = client.chat.completions.create(
    messages=input_msg,
    model='gpt-4',
    temperature=0,
    max_tokens=1,
    stop=None,
    logprobs=True,
    top_logprobs=2,
)

pprint(completion)

message = completion.choices[0].message
response = unicodedata.normalize('NFKC', message.content)
print()
print(response)

# import requests

# url = "https://lonlie.plus7.plus/dashboard/billing/usage"

# payload = {}
# headers = {
#   'Authorization': 'Bearer sk-rg8izdOSsK2tLXLt124865B116214e5bAa98C0Bc0c0b6aBc'
# }

# response = requests.request("GET", url, headers=headers, data=payload)

# print(response.text)