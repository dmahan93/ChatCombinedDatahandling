import json

import datasets
import transformers
import json
import tqdm


CHAT_SYSTEM_PROMPT ="""<|SYSTEM|># StableAssistant
- StableAssistant is A helpful and harmless Open Source AI Language Model developed by Stability and CarperAI.
- StableAssistant is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableAssistant is more than just an information source, StableAssistant is also able to write poetry, short stories, and make jokes.
- StableAssistant will refuse to participate in anything that could harm a human."""


if __name__ == '__main__':
    data = datasets.load_dataset("nomic-ai/gpt4all-j-prompt-generations")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "stablelm_tokenizer"
    )
    count = 0
    output_list = list()
    for item in tqdm.tqdm(data['train']):
        system_info = "<|SYSTEM|>You are a helpful AI designed to answer questions.<|USER|>" + item['prompt']
        user_string = ""
        if "</s>" in item['response']:
            response_items = item['response'].split("</s> \n \n")
            assist_string = "<|ASSISTANT|>" + response_items[0].strip()
            if len(response_items) > 1:
                system_info = CHAT_SYSTEM_PROMPT + "<|USER|>" + item['prompt']
            for response in response_items[1:]:
                user_msg = response.split('\n')[0]
                assist_msg = '\n'.join(response.split('\n')[1:])
                assist_string += "<|USER|>" + user_msg.strip() + "<|ASSISTANT|>" + assist_msg.strip()
        else:
            assist_string = "<|ASSISTANT|>" + item['response'].strip()
        output_string = system_info + user_string + assist_string
        output_list.append(output_string)
    with open("gpt4all-j-data.json", 'w', encoding='latin1') as f:
        json.dump(output_list, f, indent=2)