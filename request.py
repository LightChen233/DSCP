'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-12-18 14:54:57
LastEditTime: 2024-01-15 15:47:54
Description: 

'''

import asyncio
from functools import partial
import os
import fire

from utils.common_tool import read_jsonl
from utils.request_tool import request_LLM


def create_prompt(data, dataset_type, prompt_type):
    data = " ".join(data["text"])
    if dataset_type == "mix-snips":
        label_str = "[AddToPlaylist; BookRestaurant; GetWeather; PlayMusic; RateBook; SearchCreativeWork; SearchScreeningEvent]"
    else:
        label_str = "[atis_flight ; atis_airfare ; atis_ground_service ; atis_abbreviation ; atis_airline ; atis_quantity ; atis_aircraft ; atis_flight_time ; atis_city ; atis_capacity ; atis_airport ; atis_flight_no ; atis_distance ; atis_meal ; atis_ground_fare ; atis_restriction ; atis_cheapest]"
    if prompt_type == "dscp":
        prompt = f"""Assuming you are a professional multi-intent annotator, you need to label the single or multiple intents contained in the sentence based on the given one.

You need to select the intent of the sentence from the following intent list: {label_str}

You must follow these steps to gradually solve the problem of multi-intents annotation:
1. Firstly, you need to divide the sentence into multiple parts that contain different intents;
2. Secondly, you need to consider what intents each part contains;
3. Finally, you need to consider all the intents together and output them in the form of 'Result=INTENT1#INTENT2...'. Please note that there are generally only one to three multi-intentions. Please remove redundant intentions.

Here is the the sentence:\n{data}"""
    elif prompt_type == "inter-dscp":
        prompt = [f"""Assuming you are a professional multi-intent annotator, you need to label the single or multiple intents contained in the sentence based on the given one.

You need to select the intent of the sentence from the following intent list: {label_str}

You must follow these steps to gradually solve the problem of multi meaning icon annotation:
Firstly, you need to divide the sentence into multiple parts that contain different intents;

Here is the the sentence: 
{data}""",
"""Secondly, Based on the sentences you previously segmented, you need to consider what intents each part contains.""",
"""Finally, you need to consider all the intents together and output them in the form of 'Result=INTENT1#INTENT2...'\nPlease note that there are generally only one to three multi-intentions. Please remove redundant intentions."""
]
    return prompt


class MyData:
    def __init__(self, load_path) -> None:
        self.data = read_jsonl(load_path)
        for i, x in enumerate(self.data):
            self.data[i]["index"] = str(i)

def run(
        dataset_type="mix-snips",
        prompt_type="dscp",
        model_type="gpt",
        model_name="gpt-3.5-turbo",
        api_key="sk-xxx",
        request_proxy=None,
    ):
    dataset_path = "data/MixATIS/test.jsonl" if dataset_type == "mix-atis" else "data/MixSNIPS/test.jsonl"
    if not os.path.exists("./outputs"):
        os.makedirs("./outputs",exist_ok=True)
    asyncio.run(request_LLM(
        dataset=MyData(dataset_path),
        save_path=f"./outputs/{dataset_type}-{model_name}-{prompt_type}.jsonl",
        consumer_size=25,
        create_prompt_fn=partial(create_prompt, dataset_type=dataset_type, prompt_type=prompt_type),
        model_type=model_type,
        model_name=model_name,
        api_key=api_key,
        enable_multi_turn = False,
        request_proxy=request_proxy
        ))

if __name__ == "__main__":
    fire.Fire(run)
