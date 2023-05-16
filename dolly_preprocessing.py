import jsonlines
import json


SYSTEM_MESSAGES = {
    'summarization': "<|SYSTEM|>You are to summarize a passage, which will begin in the next message, along with the user's request.",
    'closed_qa': "<|SYSTEM|>You are to answer a user's question, which will begin in the next message.",
    'open_qa': "<|SYSTEM|>You are to answer a user's question, which will begin in the next message.",
    'creative_writing': "<|SYSTEM|>You are to perform some creative writing, which will begin in the next message.",
    'information_extraction': "<|SYSTEM|>You are to extract some information from a passage, which will begin in the next message.",
    'general_qa': "<|SYSTEM|>You are to answer a user's question, which will begin in the next message.",
    'classification': "<|SYSTEM|>You are to classify some text from the user's request, which will begin in the next message.",
    'brainstorming': "<|SYSTEM|>You are to brainstorm with a user."
}


if __name__ == '__main__':
    out_data = list()
    categories = list()
    with jsonlines.open(r"C:\Users\dmaha\Downloads\databricks-dolly-15k.jsonl") as reader:
        for obj in reader:
            # print(obj)
            # if obj['category'] not in set(categories):
            #     print(obj)
            # categories.append(obj['category'])
            if (obj['category'] == 'summarization') or (obj['category'] == 'information_extraction'):
                # Don't need to care about qa
                out_data.append(f"{SYSTEM_MESSAGES[obj['category']]}<|USER|>{obj['instruction']}\n{obj['context']}"
                                f"<|ASSISTANT|>{obj['response']}")
            else:
                out_data.append(f"{SYSTEM_MESSAGES[obj['category']]}<|USER|>{obj['instruction']}"
                                f"<|ASSISTANT|>{obj['response']}")
    with open("dolly2.json", 'w') as f:
        json.dump(out_data, f)