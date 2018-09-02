# encoding = utf-8
# author = xy

import torch
import jieba
import numpy as np
import pickle


def pad(data_array, length):
    """ padding """
    tmp = []
    for d in data_array:
        if len(d) > length:
            tmp.append(d[: length])
        elif len(d) < length:
            tmp.append(d + [0]*(length-len(d)))
        else:
            tmp.append(d)
    data_array = tmp
    return data_array


def deal_batch_original(batch):
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


def deal_batch(batch):
    """
    deal batch: cuda, cut
    :param batch:[content_index, content_flag, content_is_in_title, content_is_in_question, question_index,
    question_flag, start, end] or [content_index, content_flag, content_is_in_title, content_is_in_question,
    question_index, question_flag]
    :return: batch_done
    """
    def cut(data):
        max_len = get_mask(data[0]).sum(dim=1).max().item()
        max_len = int(max_len)
        data = [d[:, :max_len] for d in data]
        return data

    def padding(data, length):
        cur_len = data[0].size(1)
        if cur_len > length:
            data = [d[:, :length] for d in data]
        elif cur_len < length:
            pad_len = length - cur_len
            batch_size = data[0].size(0)
            pad_zeros = data[0].new_zeros(batch_size, pad_len)
            data = [torch.cat([d, pad_zeros], dim=1) for d in data]

        return data

    contents = batch[: 3]
    questions = batch[3: 6]
    is_training = True if len(batch) == 8 else False

    # cuda
    contents = [c.cuda() for c in contents]
    questions = [q.cuda() for q in questions]
    if is_training:
        starts = batch[6].cuda()
        ends = batch[7].cuda()

    # cut
    # contents = cut(contents)
    # questions = cut(questions)

    # padding
    # contents = padding(contents, 500)
    # questions = padding(questions, 150)

    if is_training:
        return [*contents, *questions, starts, ends]
    else:
        return [*contents, *questions]


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


def answer_search(answer_prop):
    """
     global search best answer for model predict
    :param answer_prop: (2, batch_size, c_len)
    :return: ans_s, ans_e
    """
    batch_size = answer_prop.size(1)
    c_len = answer_prop.size(2)

    ans_s_p = answer_prop[0]
    ans_e_p = answer_prop[1]
    b_zero = answer_prop.new_zeros(batch_size, 1)

    ans_s_e_p_lst = []
    for i in range(c_len):
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


def softmax(weight):
    exp = np.exp(weight)
    return exp / exp.sum()


def _rouge_score(start_y, end_y, start_pred, end_pred, gamma):
    """ 计算给定区间的(1-rouge)  """
    start = max(start_y, start_pred)
    end = min(end_y, end_pred)

    interval = end - start + 1
    if interval <= 0:
        return 1
    else:
        length_pred = end_pred - start_pred + 1
        length_y = end_y - start_y + 1
        prec = interval / length_pred if length_pred > 0 else 0
        rec = interval / length_y if length_y >0 else 0

        if prec != 0 and rec != 0:
            score = 1 - ((1 + gamma**2) * prec * rec) / (rec + gamma**2 * prec)
        else:
            score = 1
        return score


def rouge_scores(start_y, end_y, start_pro, end_pro, gamma):
    """ 计算某一条记录的期望rouge """
    result = 0
    for s in range(end_y+1):
        for j in range(start_y, len(start_pro)):
            result += _rouge_score(start_y, end_y, s, j, gamma) * start_pro[s] * end_pro[j]

    return result


def deal_data(data1, data2):
    """
     index, tag, is_in_question
    :return:
    """
    with open('data_gen/word2tag.pkl', 'rb') as file:
        lang = pickle.load(file)

    index = []
    tag = []
    is_in_each = []
    for d1, d2 in zip(data1, data2):
        i_list = []
        tag_list = []
        is_in_lst = []
        for dd in jieba.lcut(d1, HMM=False):
            i_list.append(dd)
            tag_list.append(lang[dd] if dd in lang else '<unk>')

            if dd in d2:
                is_in_lst.append(1)
            else:
                is_in_lst.append(0)

        index.append(i_list)
        tag.append(tag_list)
        is_in_each.append(is_in_lst)

    return index, tag, is_in_each


def index_tag(tag_path, data):
    """
    将 tag 转化为 index
    :param tag_path:
    :param data:
    :return:
    """
    with open(tag_path, 'rb') as file:
        lang = pickle.load(file)

    result = []
    for d in data:
        r = [lang[dd] if dd in lang else 1 for dd in d]
        result.append(r)

    return result


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)
