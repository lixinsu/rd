# encoding = utf-8
# author = xy

import torch
import jieba
import numpy as np


def match(content, question):
    """
    shorten content based on question
    :param content: str
    :param question: str
    :return: str
    """
    question_set = set(jieba.cut(question))
    content_list = content.split('。')
    scores = []
    for c in content_list:
        if c in question and c != '' and c != ' ':
            scores.append(-1)
            continue

        c = set(jieba.cut(c))
        score = len(c & question_set)
        scores.append(score)

    for s in range(len(scores)):
        if scores[s] == -1:
            if s-1 >= 0:
                scores[s-1] *= 2
            if s+1 < len(scores):
                scores[s+1] *= 2

    best_score = np.argmax(scores)

    result = []
    if best_score-1 >= 0:
        result.append(content_list[best_score-1])
    result.append(content_list[best_score])

    return '。'.join(result)


def match_pq(p, q):
    """ find the location q in p"""
    return 0, 0


def pad(data_array):
    """ padding """
    data_len = [len(d) for d in data_array]
    max_len = max(data_len)
    data_array = [d + [0]*(max_len-len(d)) for d in data_array]
    return data_array


def deal_batch(batch):
    pass


def get_mask(tensor, padding_idx=0):
    """ get mask tensor """
    return torch.ne(tensor, padding_idx).float()
