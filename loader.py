# encoding = utf-8
# author = xy

import numpy as np
import pandas as pd
import gensim
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


def load_data(df_file, merge_name, lang):
    """
    load data from .csv
    # 1. load
    # 2. index
    # 3. padding
    return: content, question, answer_start, answer_end  (list)
    """
    # load
    df = pd.read_csv(df_file)
    content = df[merge_name].values.tolist()
    question = df['question'].values.tolist()

    if merge_name+'_answer_start' in df:
        answer_start = df[merge_name+'_answer_start'].values.tolist()
        answer_end = df[merge_name+'_answer_end'].values.tolist()

    # index
    content = [jieba.lcut(c) for c in content]
    content = [lang.words2indexes(c) for c in content]
    question = [jieba.lcut(q) for q in question]
    question = [lang.words2indexes(q) for q in question]

    # padding
    content = utils.pad(content)
    question = utils.pad(question)

    if merge_name+'_answer_start' in df:
        return [content, question, answer_start, answer_end]
    else:
        return [content, question]


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













