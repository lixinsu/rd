# encoding = utf-8
# author = xy

import torch
from torch import nn
import torch.nn.functional as f


class BoundaryPointer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BoundaryPointer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.left_ptr = UniBoundaryPointer(
            input_size=input_size,
            hidden_size=hidden_size
        )

    def forward(self, hr, content_mask):
        """
        :param hr: tensor (p_seq_len, batch_size, hidden_size(*2))
        :param content_mask: tensor (batch_size, p_seq_len)
        :return: answer_range, tensor (2, batch_size, content_len)
        """
        left_range = self.left_ptr(hr, content_mask)

        return left_range


class UniBoundaryPointer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(UniBoundaryPointer, self).__init__()

        self.input_size = input_size
        self.hidden_size =hidden_size

        self.rnn_cell = nn.LSTMCell(
            input_size=input_size,
            hidden_size=hidden_size
        )

        self.vh = nn.Linear(hidden_size, hidden_size)
        self.wh = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, hr, content_mask):
        """
        :param hr: tensor (p_seq_len, batch_size, hidden_size(*2))
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
            h_k = self.rnn_cell(h_beta, h[t])
            h.append(h_k)

        answer_range = torch.stack(answer_range)

        # add 1e-6, and no gradient explosion
        content_mask = content_mask.float()
        new_mask = (content_mask - 1) * (-1e-6)
        answer_range = answer_range + new_mask.unsqueeze(0)

        return answer_range
