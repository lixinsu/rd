# coding = utf-8
# author = xy

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, mode, input_size, hidden_size, dropout_p, bidirectional, layer_num):
        super(Encoder, self).__init__()

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.direction_num = bidirectional
        self.layer_num = layer_num

        if mode == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=layer_num,
                bidirectional=bidirectional,
                dropout=dropout_p if layer_num > 1 else 0
            )
        elif mode == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=layer_num,
                bidirectional=bidirectional,
                dropout=dropout_p if layer_num > 1 else 0
            )

        self.layer_norm = nn.LayerNorm(input_size)
        self.drop = nn.Dropout(p=dropout_p)

    def forward(self, vec, mask):
        """
        :param vec: tensor (seq_len, batch_size, input_size)
        :param mask: tensor (batch_size, seq_len)
        :return: outputs: tensor (seq_len, batch_size, hidden_size)
        """
        # layer normalization
        if False:
            seq_len, batch_size, input_size = vec.shape
            vec = vec.view(-1, input_size)
            vec = self.layer_norm(vec)
            vec = vec.view(seq_len, batch_size, input_size)

        # rnn, no dropout, not state
        outputs, _ = self.rnn(vec, None)

        return outputs




if __name__ == '__main__':
    a = torch.Tensor([[1,2,3], [4,5,6]])
    print(a.sum(dim=1))
























