# encoding = utf-8
# author = xy

import numpy as np
import pandas as pd
import gensim
import torch
from torch.utils import data
import vocab
import utils


def load_vocab(vocab_path):
    """ load vocab """
    lang = vocab.Vocab()
    lang.load(vocab_path)
    return lang


def load_w2v(embedding_path, lang):
    """ load embedding vector """
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path)
    embedding = np.random.normal(size=(len(lang.w2i), model.wv.vector_size))
    for k, v in lang.w2i.items():
        if k in model.wv:
            embedding[v] = model.wv[k]
    return embedding


def load_data(df_done):
    """
    load data from df
        1. load: ['content_index', 'quetion_index', 'answer_start', 'answer_end']
        2. padding
    return: content, question, answer_start, answer_end  (list)
    """
    df = pd.read_csv(df_done)
    # 1. load
    content = df['content_index'].values.tolist()
    content = [[int(cc) for cc in c[1:-1].split(',')] for c in content]
    question = df['question_index'].values.tolist()
    question = [[int(qq) for qq in q[1:-1].split(',')] for q in question]
    answer_start = df['answer_start'].values.tolist()
    answer_end = df['answer_end'].values.tolist()

    # 2. padding
    content = utils.pad(content)
    question = utils.pad(question)

    return [content, question, answer_start, answer_end]


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













