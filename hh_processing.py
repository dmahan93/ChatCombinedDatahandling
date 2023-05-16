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
    data = datasets.load_dataset("Dahoas/full-hh-rlhf")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "stablelm_tokenizer"
    )
    count = 0
    output_list = list()
    for item in tqdm.tqdm(data['train']):
        system_info = CHAT_SYSTEM_PROMPT
        prompts = item['prompt'].strip()
        prompts = prompts.split('\n\n')
        output_string = ""
        for i, prompt in enumerate(prompts):
            if i % 2 == 0:
                output_string += "<|USER|>" + prompt.replace("Human: ", "")
            else:
                if i != len(prompts)-1:
                    output_string += "<|ASSISTANT|>" + prompt.replace("Assistant: ", "")
                else:
                    output_string += "<|ASSISTANT|>" + item['response'].strip()
        output_string = system_info + output_string
        output_list.append(output_string)
    with open("hh-data.json", 'w', encoding='latin1') as f:
        json.dump(output_list, f, indent=2)