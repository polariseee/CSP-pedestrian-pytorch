import torch
import torch.nn as nn
import pdb


# def smooth_l1_loss(pred, target):
#     absolute_loss = torch.abs(target[:, 0, :, :] - pred[:, 0, :, :]) / (target[:, 0, :, :] + 1e-10)
#     square_loss = 0.5 * ((target[:, 0, :, :] - pred[:, 0, :, :]) / (target[:, 0, :, :] + 1e-10)) ** 2
#     l1_loss = target[:, 1, :, :] * torch.where(absolute_loss.lt(1.0), square_loss, absolute_loss - 0.5)
#     assigned_boxes = target[:, 1, :, :].sum()
#     reg_loss = l1_loss.sum() / torch.Tensor([max(1.0, assigned_boxes.item())]).cuda()
#     return reg_loss


class RegrHLoss(nn.Module):
    def __init__(self, loss_weight):
        super(RegrHLoss, self).__init__()
        self.loss_weight = loss_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred, target):
        loss = target[:, 1, :, :] * self.smooth_l1(pred[:, 0, :, :], target[:, 0, :, :])
        pos_h_nums = max(1.0, torch.sum(target[:, 1, :, :]).item())
        loss_regrh = self.loss_weight*torch.sum(loss) / pos_h_nums
        return loss_regrh
