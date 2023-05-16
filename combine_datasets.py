import datasets
import json
from random import shuffle

if __name__ == '__main__':
    with open("alpaca-gpt4-data.json", encoding='latin1') as f:
        alpaca_list = json.load(f)

    with open("gpt4all-j-data.json", encoding='latin1') as f:
        gpt4all = json.load(f)
    with open("dolly2.json", encoding='latin1') as f:
        dolly = json.load(f)
    with open("sharegpt_split.json", encoding='latin1') as f:
        sharegpt_list = json.load(f)
    with open("hh-data.json", encoding='latin1') as f:
        hh = json.load(f)
    total = alpaca_list + gpt4all + dolly + sharegpt_list + hh
    shuffle(total)
    dataset = datasets.Dataset.from_dict({'text': total}).push_to_hub('ChatCombined')

    # combined = alpaca_list + sharegpt_list
    # shuffle(combined)
    # dataset = datasets.Dataset.from_dict({'text': combined}).push_to_hub('InstructFollowing')