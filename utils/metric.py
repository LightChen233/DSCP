'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-08-31 12:45:37
Description: Metric calculation class

'''
from collections import Counter
from typing import List, Dict

import numpy as np
from sklearn.metrics import f1_score

### Modified from OPENSLU


class Evaluator(object):
    """Evaluation metric funtions library class
        supported metric:
        - slot_f1
        - intent_acc
        - exactly_match_accuracy
        - intent_f1 (defult "macro_intent_f1")
            - macro_intent_f1
            - micro_intent_f1=
    """
    def __init__(self, intent_list) -> None:
        intent_list.sort()
        self.intent_dict = {x: i for i, x in enumerate(intent_list)}
        self.num_intent = len(intent_list)

    
    def intent_accuracy(self, pred_list: List, real_list: List) -> float:
        """Get  intent accuracy measured by predictions and ground-trues. Support both multi intent and single intent.

        Args:
            pred_list (List): predicted intent list
            real_list (List): golden intent list

        Returns:
            float: intent accuracy score
        """
        total_count, correct_count = 0.0, 0.0
        for p_intent, r_intent in zip(pred_list, real_list):
            if isinstance(p_intent, list):
                p_intent, r_intent = set(p_intent), set(r_intent)
            if p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0
        if total_count == 0:
            return 0
        return 1.0 * correct_count / total_count

    
    def intent_f1(self, pred_list: List[List[int]], real_list: List[List[int]], average='macro') -> float:
        """Get  intent accuracy measured by predictions and ground-trues. Support both multi intent and single intent.
        (Only support multi intent now, but you can use [[intent1], [intent2], ...] to compute intent f1 in single intent)
        Args:
            pred_list (List[List[int]]): predicted multi intent list.
            real_list (List[List[int]]): golden multi intent list.
            num_intent (int)
            average (str): support "micro" and "macro"

        Returns:
            float: intent accuracy score
        """
        
        if isinstance(real_list[0][0], str):
            real_res = []
            pred_res = []
            for r_l, p_l in zip(real_list, pred_list):
                temp_real_res = []
                temp_pred_res = []
                for r in r_l:
                    temp_real_res.append(self.intent_dict[r])
                for p in p_l:
                    if p == "None" or p not in self.intent_dict.keys():
                        temp_pred_res.append(-1)
                    else:
                        temp_pred_res.append(self.intent_dict[p])
                real_res.append(temp_real_res)
                pred_res.append(temp_pred_res)
            real_list = real_res
            pred_list = pred_res
        return f1_score(Evaluator.__instance2onehot(self.num_intent, real_list),
                        Evaluator.__instance2onehot(self.num_intent, pred_list),
                        average=average,
                        zero_division=0)

    @staticmethod
    def __multilabel2one_hot(labels, nums):
        res = [0.] * nums
        if len(labels) == 0:
            return res
        if isinstance(labels[0], list):
            for label in labels[0]:
                if label != -1:
                    res[label] = 1.
            return res
        for label in labels:
            if label != -1:
                res[label] = 1.
        return res

    @staticmethod
    def __instance2onehot(num_intent, data):
        res = []
        for intents in data:
            res.append(Evaluator.__multilabel2one_hot(intents, num_intent))
        return np.array(res)

    @staticmethod
    def max_freq_predict(sample):
        """Max frequency prediction.
        """
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

