# encoding = utf-8
# author = xy

import torch
import jieba
import numpy as np
import pickle


def pad(data_array):
    """ padding """
    data_len = [len(d) for d in data_array]
    max_len = max(data_len)
    data_array = [d + [0]*(max_len-len(d)) for d in data_array]
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

    contents = batch[: 4]
    questions = batch[4: 6]
    is_training = True if len(batch) == 8 else False

    # cuda
    contents = [c.cuda() for c in contents]
    questions = [q.cuda() for q in questions]
    if is_training:
        starts = batch[6].cuda()
        ends = batch[7].cuda()

    # cut
    contents = cut(contents)
    questions = cut(questions)

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


def rouge_score(pred_i, y_i):
    pred_len = len(pred_i)
    y_len = len(y_i)

    lengths = torch.zeros(pred_len+1, y_len+1).cuda()
    for i in range(1, pred_len+1):
        for j in range(1, y_len+1):
            if pred_i[i-1].item() == y_i[j-1].item():
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = torch.max(lengths[i-1][j], lengths[i][j-1])
    lcs = lengths[pred_len, y_len]

    prec = lcs / pred_len if pred_len > 0 else 0
    rec = lcs / y_len if y_len > 0 else 0

    if prec.item() != 0 and rec.item() != 0:
        score = ((1+1.2**2) * prec * rec) / (rec + 1.2**2 * prec)
    else:
        score = torch.tensor(1e-12).cuda()

    score = torch.log(score)
    return score


def deal_content(content, question):
    """
     index, tag, is_title, is_in_question
    :param content:
    :param question:
    :return:
    """
    with open('data_gen/word2tag.pkl', 'rb') as file:
        lang = pickle.load(file)

    index = []
    tag = []
    is_title = []
    is_in_question = []
    for c, q in zip(content, question):
        i_list = []
        tag_list = []
        is_title_list = []
        is_in_question_lst = []
        flag = True
        for cc in jieba.lcut(c, HMM=False):
            i_list.append(cc)
            tag_list.append(lang[cc] if cc in lang else '<unk>')

            if cc == '。':
                flag = False
            if flag:
                is_title_list.append(1)
            else:
                is_title_list.append(0)

            if cc in q:
                is_in_question_lst.append(1)
            else:
                is_in_question_lst.append(0)

        index.append(i_list)
        tag.append(tag_list)
        is_title.append(is_title_list)
        is_in_question.append(is_in_question_lst)

    return index, tag, is_title, is_in_question


def deal_question(question):
    """
    index, tag
    :param question:
    :return:
    """
    with open('data_gen/word2tag.pkl', 'rb') as file:
        lang = pickle.load(file)

    index = []
    tag = []
    for q in question:
        i_list = []
        tag_list = []
        for qq in jieba.lcut(q, HMM=False):
            i_list.append(qq)
            tag_list.append(lang[qq] if qq in lang else '<unk>')
        index.append(i_list)
        tag.append(tag_list)
    return index, tag


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
        r = [lang[dd] for dd in d]
        result.append(r)

    return result







