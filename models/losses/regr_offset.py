import torch
import torch.nn as nn
import pdb


# def smooth_l1_loss(pred, target):
#     absolute_loss = torch.abs(target[:, :2, :, :] - pred[:, :, :, :])
#     square_loss = 0.5 * (target[:, :2, :, :] - pred[:, :, :, :]) ** 2
#     l1_loss = target[:, 2, :, :] * torch.sum(torch.where(absolute_loss.lt(1.0), square_loss, absolute_loss - 0.5), axis=1)
#     assigned_boxes = target[:, 1, :, :].sum()
#     reg_loss = l1_loss.sum() / torch.Tensor([max(1.0, assigned_boxes.item())]).cuda()
#     return reg_loss


class RegrOffsetLoss(nn.Module):
    def __init__(self, loss_weight):
        super(RegrOffsetLoss, self).__init__()
        self.loss_weight = loss_weight
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred, target):
        # loss_regrh = self.loss_weight*smooth_l1_loss(pred, target)
        smooth_l1_loss = target[:, 2, :, :].unsqueeze(dim=1)*self.smoothl1(pred, target[:, :2, :, :])
        pos_offset_nums = max(1.0, torch.sum(target[:, 2, :, :]).item())
        loss_regrh = self.loss_weight * torch.sum(smooth_l1_loss) / pos_offset_nums
        return loss_regrh
