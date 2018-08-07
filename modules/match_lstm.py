# encoding = utf-8
# author = xy


import torch
from torch import nn
from modules.layers import embedding
from modules.layers import encoder
from modules.layers import match_rnn
from modules.layers import pointer
import utils


class Model(nn.Module):
    """ match-lstm model for machine comprehension"""
    def __init__(self, param):
        """
        :param param: embedding, hidden_size, dropout_p, encoder_dropout_p, encoder_direction_num, encoder_layer_num
        """
        super(Model, self).__init__()

        self.w2v_size = param['embedding'].shape[1]
        self.vocab_size = param['embedding'].shape[0]
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_bidirectional = param['encoder_bidirectional']
        self.encoder_layer_num = param['encoder_layer_num']

        self.dropout = nn.Dropout(p=self.dropout_p)

        if param['embedding_type'] == 'standard':
            self.embedding = embedding.Embedding(param['embedding'])

        self.encoder = encoder.Encoder(
            mode=param['encoder_mode'],
            input_size=self.embedding.embedding_dim,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=self.encoder_bidirectional,
            layer_num=self.encoder_layer_num
        )

        self.match_rnn = match_rnn.MatchRNN(
            input_size=self.hidden_size*2 if self.encoder_bidirectional else self.hidden_size,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p
        )

        self.pointer_net = pointer.BoundaryPointer(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size
        )

    def forward(self, batch):
        """
        :param batch: [content, question, answer_start, answer_end]
        :return: ans_range (batch_size, 2, context_len)
        """
        content = batch[0]
        question = batch[1]
        answer_start = batch[2]
        answer_end = batch[3]

        # embedding
        content_mask = utils.get_mask(content)  # (batch_size, seq_len)
        question_mask = utils.get_mask(question)
        content_vec = self.embedding(content)  # (seq_len, batch_size, embedding_dim)
        question_vec = self.embedding(question)

        # encoder
        content_vec = self.encoder(content_vec, content_mask)  # (seq_len, batch_size, hidden_size(*2))
        question_vec = self.encoder(question_vec, question_mask)

        # match-rnn
        hr = self.match_rnn(content_vec, content_mask, question_vec, question_mask)  # (p_seq_len, batch_size, hidden_size(*2))

        # pointer
        ans_range = self.pointer_net(hr, content_mask)

        return ans_range,











