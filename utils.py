# encoding = utf-8
# author = xy

import torch
import jieba
import numpy as np


def pad(data_array):
    """ padding """
    data_len = [len(d) for d in data_array]
    max_len = max(data_len)
    data_array = [d + [0]*(max_len-len(d)) for d in data_array]
    return data_array


def deal_batch(batch):
    """
    deal batch: cuda, cut
    :param batch:[content, question, start, end] or [content, question]
    :return: batch_done
    """
    def cut(indexs):
        max_len = get_mask(indexs, 0).sum(dim=1).max().item()
        return indexs[:, :max_len+1]

    is_training = True if len(batch) == 4 else False
    if is_training:
        contents, questions, starts, ends = batch
    else:
        contents, questions = batch

    # cuda
    contents = contents.cuda()
    questions = questions.cuda()
    if is_training:
        starts = starts.cuda()
        ends = ends.cuda()

    # cut
    contents = cut(contents)
    questions = cut(questions)

    if is_training:
        return [contents, questions, starts, ends]
    else:
        return [contents, questions]





def get_mask(tensor, padding_idx=0):
    """ get mask tensor """
    return torch.ne(tensor, padding_idx)
