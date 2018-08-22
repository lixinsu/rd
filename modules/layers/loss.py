# coding = utf-8
# author = xy


import torch
from torch.nn.modules import loss
import torch.nn.functional as f
import utils


class MyNLLLoss(loss._Loss):
    """ MLE 最大似然估计 """
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


class RougeLoss(loss._Loss):
    """ MRT 最小风险 """
    def __init__(self, lam):
        super(RougeLoss, self).__init__()

        self.lam = lam

        self.mle = MyNLLLoss()

    def forward(self, outputs, batch):
        """
        :param outputs: tensor (2, batch_size, content_seq_len)
        :param batch: tensor (content, question, start, end)
        :return: loss
        """
        # mrt
        start_y = batch[2]
        end_y = batch[3]
        y = [torch.range(s, e).cuda() for s, e in zip(start_y, end_y)]

        start_pred, end_pred = torch.max(outputs, dim=2)[1]
        pred = [torch.range(s, e).cuda() if s <= e else torch.Tensor([-1]).cuda() for s, e in zip(start_pred, end_pred)]

        loss_mrt = [utils.rouge_score(pred_i, y_i) for pred_i, y_i in zip(pred, y)]
        loss_mrt_tmp = 0
        for i in loss_mrt:
            loss_mrt_tmp += i
        loss_mrt = loss_mrt_tmp / len(loss_mrt)

        # mle
        loss_mle = self.mle(outputs, batch)

        loss_value = loss_mle + self.lam * loss_mrt
        return loss_value
