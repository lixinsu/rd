# encoding = utf-8
# author = xy

import numpy as np
import pandas as pd
import jieba
import torch
from torch.utils import data
import vocab
import utils


def load_vocab(vocab_path):
    """ load vocab """
    lang = vocab.Vocab()
    lang.load(vocab_path)
    return lang


def load_w2v(embedding_path):
    """ load embedding vector """
    embedding_np = np.load(embedding_path)
    return embedding_np


def load_data_orignal(df_file, lang):
    """
    load data from .csv
    # 1. load
    # 2. index
    # 3. padding
    return: content, question, answer_start, answer_end  (list)
    """
    # load
    df = pd.read_csv(df_file)
    content = df['merge'].values.tolist()
    question = df['question'].values.tolist()

    if 'answer_start' in df:
        answer_start = df['answer_start'].values.tolist()
        answer_end = df['answer_end'].values.tolist()

    # index
    content = [jieba.lcut(c, HMM=False) for c in content]
    content = [lang.words2indexes(c) for c in content]
    question = [jieba.lcut(q, HMM=False) for q in question]
    question = [lang.words2indexes(q) for q in question]

    # padding
    content = utils.pad(content)
    question = utils.pad(question)

    if 'answer_start' in df:
        return [content, question, answer_start, answer_end]
    else:
        return [content, question]


def load_data(df_file, lang):
    """
    load data from .csv
    # 1. load
    # 2. index, tag(词性), 是否在答案中出现， 是否是标题
    # 3. padding
    return: content, question, answer_start, answer_end  (list)
    """

    # load
    df = pd.read_csv(df_file)
    content = df['merge'].values.tolist()
    question = df['question'].values.tolist()

    if 'answer_start' in df:
        answer_start = df['answer_start'].values.tolist()
        answer_end = df['answer_end'].values.tolist()

    # content: index, flag, is_title, is_in_question
    content_index, content_flag, content_is_in_question = utils.deal_data(content, question)
    content_index = [lang.words2indexes(c) for c in content_index]
    content_flag = utils.index_tag('data_gen/tag2index.pkl', content_flag)

    # question: index, flag
    question_index, question_flag, question_is_in_content = utils.deal_data(question, content)
    question_index = [lang.words2indexes(q) for q in question_index]
    question_flag = utils.index_tag('data_gen/tag2index.pkl', question_flag)

    # padding
    content_index = utils.pad(content_index, 500)
    content_flag = utils.pad(content_flag, 500)
    content_is_in_question = utils.pad(content_is_in_question, 500)

    question_index = utils.pad(question_index, 150)
    question_flag = utils.pad(question_flag, 150)
    question_is_in_content = utils.pad(question_is_in_content, 150)

    if 'answer_start' in df:
        return [content_index, content_flag, content_is_in_question, question_index, question_flag,
                question_is_in_content, answer_start, answer_end]
    else:
        return [content_index, content_flag, content_is_in_question, question_index, question_flag, question_is_in_content]


def build_loader(dataset, batch_size, shuffle, drop_last):
    """
    build data loader
    return: a instance of Dataloader
    """
    dataset = [torch.LongTensor(d) for d in dataset]
    dataset = data.TensorDataset(*dataset)
    data_iter = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return data_iter













