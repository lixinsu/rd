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
        max_len = get_mask(indexs).sum(dim=1).max().item()
        max_len = int(max_len)
        return indexs[:, :max_len]

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
    return torch.ne(tensor, padding_idx).float()


def masked_flip(seq_tensor, mask):
    """
     flip seq_tensor
    :param seq_tensor: (seq_len, batch_size, input_size)
    :param mask: (batch_size, seq_len)
    :return: (seq_len, batch_size, input_size)
    """
    length = mask.eq(1).long().sum(dim=1)
    batch_size = seq_tensor.size(1)

    outputs = []
    for i in range(batch_size):
        temp = seq_tensor[:, i, :]
        temp_length = length[i]

        idx = list(range(temp_length-1, -1, -1)) + list(range(temp_length, seq_tensor.size(0)))
        idx = seq_tensor.new_tensor(idx, dtype=torch.long)

        temp = temp.index_select(0, idx)
        outputs.append(temp)

    outputs = torch.stack(outputs, dim=1)
    return outputs


def answer_search(answer_prop, mask, max_tokens):
    """
     global search best answer for model predict
    :param answer_prop: (2, batch_size, c_len)
    :param mask: (batch_size, c_len)
    :param max_tokens: .
    :return: ans_range, score
    """
    batch_size = answer_prop.size(1)
    c_len = answer_prop.size(2)

    # get min length
    lengths = mask.data.eq(1).long().sum(dim=1).squeeze()
    min_length, _ = torch.min(lengths, 0)
    min_length = min_length.item()

    # max move steps
    max_move = max_tokens + c_len - min_length
    max_move = min(c_len, max_move)

    ans_s_p = answer_prop[0]
    ans_e_p = answer_prop[1]
    b_zero = answer_prop.new_zeros(batch_size, 1)

    ans_s_e_p_lst = []
    for i in range(max_move):
        temp_ans_s_e_p = ans_s_p * ans_e_p
        ans_s_e_p_lst.append(temp_ans_s_e_p)

        ans_s_p = ans_s_p[:, :(c_len - 1)]
        ans_s_p = torch.cat([b_zero, ans_s_p], dim=1)

    ans_s_e_p = torch.stack(ans_s_e_p_lst, dim=2)

    # get the best end position, and move steps
    max_prop1, max_prop_idx1 = torch.max(ans_s_e_p, 1)
    max_prop2, max_prop_idx2 = torch.max(max_prop1, 1)

    ans_e = max_prop_idx1.gather(1, max_prop_idx2.unsqueeze(1)).squeeze(1)
    ans_s = ans_e - max_prop_idx2

    return ans_s, ans_e
































