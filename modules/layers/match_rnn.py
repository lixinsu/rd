# encoding = utf-8
# author = xy

import utils
import torch
from torch import nn
import torch.nn.functional as f


class MatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p):
        super(MatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.right_match = UniMatchRNN(input_size, hidden_size)
        self.left_match = UniMatchRNN(input_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, content_vec, content_mask, question_vec, question_mask):
        """
        :param content_vec: tensor (p_seq_len, batch_size, input_size)
        :param content_mask: tensor (batch_size, p_seq_len)
        :param question_vec: tensor (q_seq_len, batch_size, input_size)
        :param question_mask: tensor (batch_size, q_seq_len)
        :return: result (p_seq_len, batch_size, hidden_size*2))
        """
        content_vec = self.dropout(content_vec)
        question_vec = self.dropout(question_vec)

        right_result = self.right_match(content_vec, question_vec, question_mask)
        content_vec_clip = utils.masked_flip(content_vec, content_mask)
        left_result = self.left_match(content_vec_clip, question_vec, question_mask)
        left_result = utils.masked_flip(left_result, content_mask)

        result = torch.cat([right_result, left_result], dim=2)  # (p_seq_len, batch_size, hidden_size*2)
        result = content_mask.transpose(0, 1).unsqueeze(2) * result

        return result


class UniMatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UniMatchRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn_cell = nn.LSTMCell(
            input_size=input_size*2,
            hidden_size=hidden_size
        )

        self.wq = nn.Linear(input_size, hidden_size)
        self.wp = nn.Linear(input_size, hidden_size)
        self.wr = nn.Linear(hidden_size, hidden_size)
        self.wg = nn.Linear(hidden_size, 1)

        self.layer_norm = nn.LayerNorm(input_size*2)
        self.reset_parameters()

    def reset_parameters(self):
        """ use xavier_uniform to initialize GRU/LSTM weights"""
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, content_vec, question_vec, question_mask):
        """
        :param content_vec: tensor (p_seq_len, batch_size, input_size)
        :param question_vec: tensor (q_seq_len, batch_size, input_size)
        :param question_mask: tensor (batch_size, q_seq_len)
        :return: hr, tensor (p_seq_len, batch_size, hidden_size)
        """

        batch_size = content_vec.size(1)
        p_seq_len = content_vec.size(0)

        h_0 = question_vec.new_zeros(batch_size, self.hidden_size)
        h = [(h_0, h_0)]

        wh = self.wq(question_vec)  # (q_seq_len, batch_size, hidden_size)

        for t in range(p_seq_len):
            # attention
            hp = content_vec[t]
            hp = self.wp(hp).unsqueeze(0)  # (1, batch_size, hidden_size)

            hr = h[t][0]
            hr = self.wr(hr).unsqueeze(0)  # (1, batch_size, hidden_size)

            g = f.tanh(wh + hp + hr)
            alpha = self.wg(g).squeeze(2).transpose(0, 1)  # (batch_size, q_seq_len)

            # mask
            mask = question_mask.eq(0)
            alpha.masked_fill_(mask, -float('inf'))
            alpha = f.softmax(alpha, dim=1)

            h_alpha = torch.bmm(alpha.unsqueeze(1), question_vec.transpose(0, 1)).squeeze(1)  # (batch_size, input_size)
            z = torch.cat([content_vec[t], h_alpha], dim=1)

            z = self.layer_norm(z)
            hr = self.rnn_cell(z, h[t])
            h.append(hr)

        hr = [hh[0] for hh in h[1:]]
        hr = torch.stack(hr)

        return hr
