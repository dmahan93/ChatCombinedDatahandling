import json

import datasets
import transformers
import json
import tqdm


CHAT_SYSTEM_PROMPT ="""<|SYSTEM|>You are to follow instructions faithfully from the user, which will begin in the next message."""


if __name__ == '__main__':
    data = datasets.load_dataset("c-s-ale/alpaca-gpt4-data")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "stablelm_tokenizer"
    )
    output_list = list()
    for item in tqdm.tqdm(data['train']):
        system_info = "<|SYSTEM|>You are to follow instructions faithfully from the user, which will begin in the next message.<|USER|>" + item['instruction']
        user_string = "" if item['input'].strip() == "" else "\n" + item['input']
        assist_string = "<|ASSISTANT|>" + item['output']
        output_string = system_info + user_string + assist_string
        if len(tokenizer(output_string)['input_ids']) < 4096:
            output_list.append(output_string)
    with open("alpaca-gpt4-data.json", 'w', encoding='latin1') as f:
        json.dump(output_list, f, indent=2)