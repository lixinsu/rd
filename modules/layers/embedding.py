# encoding = utf-8
# author = xy

import torch
from torch import nn
import numpy as np


class Embedding(nn.Module):
    """
    standard embedding
    input: tensor (batch_size, seq_len)
    return: tensor (seq_len, batch_size, w2v_size)
    """
    def __init__(self, embedding):
        super(Embedding, self).__init__()

        self.vocab_size = embedding.shape[0]
        self.w2v_size = embedding.shape[1]

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.w2v_size,
            padding_idx=0,
            _weight=torch.Tensor(embedding)
        )

        self.embedding_dim = self.embedding.embedding_dim

    def forward(self, tensor):
        return self.embedding(tensor).transpose(0, 1)


class ExtendEmbedding(nn.Module):
    """
    expanded embedding
    :return (seq_len, batch_size, embedding_size+5)
    """
    def __init__(self, embedding):
        super(ExtendEmbedding, self).__init__()

        self.sd_embedding = Embedding(embedding)

        self.tag_embedding = nn.Embedding(
            num_embeddings=60,
            embedding_dim=4,
            padding_idx=0
        )
        self.embedding_dim = self.sd_embedding.embedding_dim + 5

    def forward(self, data):
        word_embedding = self.sd_embedding(data[0])  # (seq_len, batch_size, embedding_size)
        tag_embedding = self.tag_embedding(data[1]).transpose(0, 1)  # (seq_len, batch_size, 4)
        is_in_embedding = data[2].transpose(0, 1).unsqueeze(2).float()  # (seq_len, batch_size, 1)

        result = torch.cat([word_embedding, tag_embedding, is_in_embedding], dim=2)

        return result
