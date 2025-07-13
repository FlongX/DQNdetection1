import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
ce_loss = CrossEntropyLoss()


def get_tp_fp_fn(predict, target):
    tp = (predict * target).sum()
    fp = (predict * (1 - target)).sum()
    fn = ((1 - predict) * target).sum()
    return tp, fp, fn


class DiceLoss(nn.Module):
    def __init__(self, n_classes=2):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice
        return loss / self.n_classes


