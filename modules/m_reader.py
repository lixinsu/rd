# coding = utf-8
# author = xy


import torch
from torch import nn
from torch.nn import functional as f
from modules.layers import embedding
from modules.layers import encoder


num_align_hops = 2
num_ptr_hops = 2


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        self.embedding_type = param['embedding_type']
        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']

        # embedding
        if self.embedding_type == 'standard':
            self.embedding = embedding.Embedding(param['embedding'])
            is_bn = False
        else:
            self.embedding = embedding.ExtendEmbedding(param['embedding'])
            is_bn = True

        # encoder
        input_size = self.embedding.embedding_dim
        self.encoder = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=True,
            layer_num=self.encoder_layer_num,
            is_bn=is_bn
        )

        # align
        self.aligner = nn.ModuleList([SeqToSeqAtten() for _ in range(num_align_hops)])
        self.aligner_sfu = nn.ModuleList([SFU() for _ in range(num_align_hops)])

        # self align
        self.self_aligner = nn.ModuleList([SeqToSeqAtten() for _ in range(num_align_hops)])
        self.self_aligner_sfu = nn.ModuleList([SFU() for _ in range(num_align_hops)])

        # aggregation
        self.aggregation = nn.ModuleList([encoder.Rnn() for _ in range(num_align_hops)])

        # pointer
        self.ptr_net = nn.ModuleList([Pointer() for _ in range(num_ptr_hops)])




    def forward(self):
        pass




class SeqToSeqAtten(nn.Module):
    def __init__(self):
        super(SeqToSeqAtten, self).__init__()

    def forward(self, content_vec, question_vec, question_mask):
        """
        :param content_vec: (c_len, batch_size, hidden_size)
        :param question_vec:
        :param question_mask:
        :return: (c_len, batch_size, hidden_size)
        """
        content_vec = content_vec.transpose(0, 1)  # (batch_size, c_len, hidden_size)
        question_vec = question_vec.transpose(0, 1)

        b = torch.bmm(content_vec, question_vec.transpose(1, 2))  # (batch_size, c_len, q_len)

        # mask
        mask = question_mask.eq(0).unsqueeze(1).expand(b.size())  # (batch_size, c_len, q_len)
        b.masked_fill_(mask, -float('inf'))

        b = f.softmax(b, dim=2)
        q = torch.bmm(b, question_vec)  # (batch_size, c_len, hidden_size)
        q = q.transpose(0, 1)  # (c_len, batch_size, hidden_size)

        return q


class SFU():
    pass


class Pointer():
    pass

















