import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def csp_center_loss(pred, target, beta, gamma):
    pred_sigmoid = torch.sigmoid(pred)
    positives = target[:, 2, :, :]
    negatives = target[:, 1, :, :] - target[:, 2, :, :]
    foreground_weight = positives * (1.0 - pred_sigmoid[:, 0, :, :]) ** gamma
    background_weight = negatives * ((1.0 - target[:, 0, :, :]) ** beta) * (pred_sigmoid[:, 0, :, :] ** gamma)
    focal_weight = foreground_weight + background_weight
    assigned_boxes = torch.sum(target[:, 2, :, :])
    # classification_loss = F.binary_cross_entropy_with_logits(
    #     pred[:, 0, :, :], target[:, 2, :, :], reduction='none') * focal_weight
    classification_loss = F.binary_cross_entropy(pred_sigmoid[:, 0, :, :], target[:, 2, :, :], reduction='none')
    pos_nums = max(1.0, assigned_boxes.item())
    class_loss = torch.sum(focal_weight * classification_loss) / pos_nums
    return class_loss


class CSPCenterLoss(nn.Module):
    def __init__(self,
                 beta=4,
                 gamma=2,
                 loss_weight=0.01):
        super(CSPCenterLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_cls = self.loss_weight*csp_center_loss(pred, target, self.beta, self.gamma)
        return loss_cls
