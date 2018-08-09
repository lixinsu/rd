# coding = utf-8
# author = xy


import torch
from torch.nn.modules import loss
import torch.nn.functional as f


class MyNLLLoss(loss._Loss):
    def __init__(self):
        super(MyNLLLoss, self).__init__()

    def forward(self, outputs, batch):
        """
        :param outputs: tensor (2, batch_size, content_seq_len)
        :param batch: tensor
        :return:loss
        """
        y_start = batch[2]
        y_end = batch[3]
        outputs = torch.log(outputs)
        start_loss = f.nll_loss(outputs[0], y_start)
        end_loss = f.nll_loss(outputs[1], y_end)
        return start_loss + end_loss
