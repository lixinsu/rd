# coding = utf-8
# author = xy


import torch
from torch.nn.modules import loss
import torch.nn.functional as f


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
        y_start = batch[-2]
        y_end = batch[-1]
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
        start_y = batch[-2].float()
        end_y = batch[-1].float()
        start_pred, end_pred = torch.max(outputs, dim=2)[1].float()

        start_tmp = torch.max(start_y, start_pred)
        end_tmp = torch.min(end_y, end_pred)
        tmp = end_tmp - start_tmp + 1
        tmp = torch.max(tmp, tmp.new_zeros(tmp.size()))
        mask = tmp.eq(0)

        interval_y = end_y - start_y + 1
        interval_pred = end_pred - start_pred + 1

        mask_y = (interval_y <= 0)
        interval_y.masked_fill_(mask_y, 1)
        mask_pred = (interval_pred <= 0)
        interval_pred.masked_fill_(mask_pred, 1)

        prec = tmp / interval_pred
        rec = tmp / interval_y

        score = ((1 + 1.2**2) * prec * rec) / (rec + 1.2**2*prec + 1e-6)

        score.masked_fill_(mask, 0)
        score.masked_fill_(mask_y, 0)
        score.masked_fill_(mask_pred, 0)

        score = 1 - score

        score = torch.mean(score)

        # mle
        loss_mle = self.mle(outputs, batch)
        loss_value = loss_mle + self.lam * score

        return loss_value
