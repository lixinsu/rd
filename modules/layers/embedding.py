# encoding = utf-8
# author = xy

import torch
from torch import nn


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
