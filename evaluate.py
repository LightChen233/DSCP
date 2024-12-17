'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-07-07 11:34:07
LastEditTime: 2024-01-15 17:24:40
Description: 

'''
import json
import re
import random

import fire
random.seed(42)
from utils.metric import Evaluator


def prepare_origin_data(dataset_type):
    BASE_PATH = "data/MixSNIPS/" if dataset_type == "mix-snips" else "data/MixATIS/"
    origin_data_path = BASE_PATH + "test.jsonl"
    
    origin_data = []
    origin_label_set = set()
    with open(origin_data_path, "r", encoding="utf8") as f:
        for line in f:
            sample = json.loads(line.strip())
            origin_data.append(sample)
            origin_label_set = origin_label_set | set(sample["intent"].split("#"))
    return origin_data, origin_label_set

def extract_intents(message, origin_label_set, inp):
    if message == "":
        return None, None
    if not message.endswith('.'):
        message += '.'
    message=message.replace("Result", "RESULT")
    message=message.replace("Intent", "RESULT").replace("RESULT=INTENT1#INTENT2", "").replace("#RESULT=", "#")
    message=message.replace("\_", "_")
    pred_intent = re.findall(r'RESULT=(.*?)[\.\n]', message)
    if len(pred_intent) < 1:
        pred_intent = re.findall(r'RESULT:(.*?)[\.\n]', message)
        if len(pred_intent) < 1:
            pred_intent = re.findall(r'RESULT =(.*?)[\.\n]', message)
            if len(pred_intent) < 1:
                pred_intent = ""
            else:
                pred_intent = pred_intent[-1]
        else:
            pred_intent = pred_intent[-1]
    else:
        pred_intent = pred_intent[-1]
    pred_intent = pred_intent.strip()
    pred_intent = pred_intent.strip("\"")
    pred_intent = pred_intent.strip(".")
    if pred_intent == "":
        message = re.split(r"RESULT[:= ]", message)[-1]
        for label_ in origin_label_set:
            if label_ in message:
                pred_intent += "#" + label_
    pred_intents = set([x.strip().strip("\"").strip("'") for x in re.split("[#;/]", pred_intent) if x.strip() != "" and x.strip()!= "None"])
    gold_intents = set(inp['intent'].split("#"))
    return pred_intents, gold_intents


def run(
        dataset_type="mix-snips",
        prompt_type="dscp",
        model_name="gpt35",
        load_path=None
    ):
    origin_data, origin_label_set = prepare_origin_data(dataset_type)
    if load_path is None:
        load_path = f"experiment/{model_name}/{dataset_type}/{prompt_type}.jsonl"

    request_data = []
    with open(load_path, "r", encoding="utf8") as f:
        for line in f:
            sample = json.loads(line.strip())
            request_data.append(sample)
    output_data = []
    correct = 0
    total = 0
    record_list = []
    label_set = set()
    gold_intent_res = []
    pred_intent_res = []
    for i, req in enumerate(request_data):
        
        # Extract intents
        inp = origin_data[int(req["index"])]
        if int(req["index"]) not in record_list:
            record_list.append(int(req["index"]))
        message = req["pred"][-1]["content"][0]["text"]
        pred_intents, gold_intents = extract_intents(message, origin_label_set, inp)
        
        total += 1
        if pred_intents == gold_intents:
            correct += 1
        
        label_set = label_set | gold_intents
        gold_intent_res.append(list(gold_intents))
        pred_intent_res.append(list(pred_intents))
        
    print("Correct Num. ", correct, "\t", "Total Num. ", total)
    evaluator = Evaluator(list(label_set))
    print("Intent Acc: ", evaluator.intent_accuracy(pred_intent_res, gold_intent_res))
    print("Intent Macro F1: ", evaluator.intent_f1(pred_intent_res, gold_intent_res, average="macro"))
    print("Intent Micro F1: ", evaluator.intent_f1(pred_intent_res, gold_intent_res, average="micro"))

if __name__ == "__main__":
    fire.Fire(run)