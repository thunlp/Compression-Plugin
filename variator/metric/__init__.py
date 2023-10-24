import numpy as np
import bmtrain as bmt
import torch
import re
import string
from collections import Counter
import json

def binary_f1_score(y_pred, y_true):
    tp = (y_true[y_pred == 1] == 1).sum()
    fp = (y_true[y_pred == 1] == 0).sum()
    fn = (y_true[y_pred == 0] == 1).sum()

    total_tp = bmt.sum_loss(tp)
    total_fp = bmt.sum_loss(fp)
    total_fn = bmt.sum_loss(fn)

    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)

    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)

def normalize_answer(s):
    # return s
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def squad_em(predict, answers):
    em = 0
    for pre, ans in zip(predict, answers):
        if pre in ans:
            em += 1
        # else:
        #     print("predict: %s\t answer: %s" % (pre, ans))
    return em

def squad_f1(predict, answers):
    ret = 0
    for pred, ans in zip(predict, answers):
        # if pred == "no answer":
        #     continue
        prediction_tokens = pred.split()
        cpred_token = Counter(prediction_tokens)
        curf1 = []
        for a in ans:
            ground_truth_tokens = a.split()
            common = cpred_token & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                curf1.append(0)
            else:
                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
                curf1.append(f1)
        ret += max(curf1)
    return ret


def squad_metric(predict, answers):

    pred = [normalize_answer(ans) for ans in predict]
    ground = [{normalize_answer(a) for a in json.loads(ans)} for ans in answers]

    em_sum = squad_em(pred, ground)
    f1_sum = squad_f1(pred, ground)

    ins_num = len(pred)

    # squad的生成只能在一张卡上跑，不需要全部求和
    # total_em = bmt.sum_loss(em_sum)
    # total_f1 = bmt.sum_loss(f1_sum)
    # total_ins = bmt.sum_loss(ins_num)

    # return total_em / total_ins, total_f1 / total_ins
    return em_sum / ins_num, f1_sum / ins_num