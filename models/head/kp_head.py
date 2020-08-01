import torch
import math
import torch.nn as nn
from models.py_utils import DeconvModule, L2Normalization
from models.losses import CSPCenterLoss, RegrHLoss, RegrOffsetLoss
from mmcv.cnn import ConvModule
import pdb


class KpHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level,
                 fusion_level,
                 end_level,
                 csp_center_loss=None,
                 regr_h_loss=None,
                 regr_offset_loss=None):
        super(KpHead, self).__init__()

        self.num_ins = len(in_channels)

        self.start_level = start_level
        self.fusion_level = fusion_level
        self.end_level = end_level
        self.concat_level = self.end_level - self.start_level + 1
        assert self.fusion_level >= self.start_level

        self.deconvs = nn.ModuleList()
        self.l2_norms = nn.ModuleList()

        for i in range(self.start_level, self.end_level + 1):
            l2_norm = L2Normalization(out_channels, scale=10.0)
            self.l2_norms.append(l2_norm)

        for i in range(self.fusion_level + 1, self.end_level + 1):
            if i == self.fusion_level + 1:
                stride, padding = 2, 1
            else:
                stride, padding = 4, 0
            d_conv = DeconvModule(in_channels[i],
                                  out_channels,
                                  kernel_size=4,
                                  stride=stride,
                                  padding=padding)
            self.deconvs.append(d_conv)

        self.feat = ConvModule(out_channels*self.concat_level,
                               out_channels,
                               3,
                               stride=1,
                               padding=1,
                               norm_cfg=dict(type='BN', requires_grad=True),
                               inplace=False)
        self.cls_conv = nn.Conv2d(out_channels, 1, 1)
        self.reg_conv = nn.Conv2d(out_channels, 1, 1)
        self.offset_conv = nn.Conv2d(out_channels, 2, 1)
        self.init_weights()

        self.csp_center_loss = CSPCenterLoss(**csp_center_loss)
        self.regr_loss = RegrHLoss(**regr_h_loss)
        self.offset_loss = RegrOffsetLoss(**regr_offset_loss)

    def init_weights(self):
        for m in self.feat.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

        nn.init.xavier_normal_(self.cls_conv.weight)
        nn.init.xavier_normal_(self.reg_conv.weight)
        nn.init.xavier_normal_(self.offset_conv.weight)

        nn.init.constant_(self.cls_conv.bias, -math.log(0.99/0.01))
        nn.init.constant_(self.reg_conv.bias, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == self.num_ins

        deconv_outs = []

        for i in range(self.fusion_level + 1, self.end_level + 1):
            deconv_outs.append(self.deconvs[i-1](inputs[i]))

        norm_outs = []

        for i in range(self.start_level, self.end_level + 1):
            norm_outs.append(self.l2_norms[i-1](deconv_outs[i-1]))

        cat_out = torch.cat(norm_outs, dim=1)

        cat_out = self.feat(cat_out)

        x_class = self.cls_conv(cat_out)
        x_regr = self.reg_conv(cat_out)
        x_offset = self.offset_conv(cat_out)
        return [x_class, x_regr, x_offset]

    def loss(self,
             preds,
             seman_map,
             scale_map,
             offset_map):
        cls_pred, regr_pred, offset_pred = preds[0], preds[1], preds[2]
        loss_cls = self.csp_center_loss(cls_pred, seman_map)
        loss_regr = self.regr_loss(regr_pred, scale_map)
        loss_offset = self.offset_loss(offset_pred, offset_map)
        return dict(loss_cls=loss_cls, loss_regr=loss_regr, loss_offset=loss_offset)
