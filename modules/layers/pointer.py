# encoding = utf-8
# author = xy

import torch
from torch import nn
import torch.nn.functional as f


class BoundaryPointer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p, bidirectional):
        super(BoundaryPointer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout_p
        self.bidirectional = bidirectional

        self.right_ptr = UniBoundaryPointer(
            input_size=input_size,
            hidden_size=hidden_size
        )
        if bidirectional:
            self.left_prt = UniBoundaryPointer(
                input_size=input_size,
                hidden_size=hidden_size
            )

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, hr, content_mask):
        """
        :param hr: tensor (p_seq_len, batch_size, hidden_size(*2))
        :param content_mask: tensor (batch_size, p_seq_len)
        :return: answer_range, tensor (2, batch_size, p_seq_len)
        """
        hr = self.dropout(hr)

        right_range = self.right_ptr(hr, content_mask)
        result = right_range
        if self.bidirectional:
            left_range = self.left_prt(hr, content_mask)
            left_range = left_range[[1, 0], :]
            result = (right_range + left_range) / 2

        # add 1e-6, and no gradient explosion
        content_mask = content_mask.float()
        new_mask = (content_mask - 1) * (-1e-6)
        result = result + new_mask.unsqueeze(0)

        return result


class UniBoundaryPointer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UniBoundaryPointer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn_cell = nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size
        )

        self.vh = nn.Linear(hidden_size*2, hidden_size)
        self.wh = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

        self.layer_norm = nn.LayerNorm(input_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, hr, content_mask):
        """
        :param hr: tensor (p_seq_len, batch_size, hidden_size*2)
        :param content_mask: tensor (batch_size, p_seq_len)
        :return: answer_range, tensor (2, batch_size, p_seq_len)
        """
        batch_size = hr.size(1)
        h_0 = hr.new_zeros(batch_size, self.hidden_size)
        h = [(h_0, h_0)]

        answer_range = []
        for t in range(2):
            vh = self.vh(hr)  # (p_seq_len, batch_size, hidden_size)
            wh = self.wh(h[t][0]).unsqueeze(0)  # (1, batch_size, hidden_size)
            fk = f.tanh(vh + wh)
            vf = self.v(fk).squeeze(2).transpose(0, 1)  # (batch_size, p_seq_len)

            # mask
            mask = content_mask.eq(0)
            vf.masked_fill_(mask, -float('inf'))
            beta = f.softmax(vf, dim=1)

            answer_range.append(beta)

            h_beta = torch.bmm(beta.unsqueeze(1), hr.transpose(0, 1)).squeeze(1)  # (batch_size, hidden_size)
            h_beta = self.layer_norm(h_beta)
            h_k = self.rnn_cell(h_beta, h[t])
            h.append(h_k)

        answer_range = torch.stack(answer_range)

        return answer_range
