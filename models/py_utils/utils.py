import torch
import numpy as np
from collections import OrderedDict
from external import NMS
import pdb


def parse_losses(losses):
    log_vars = OrderedDict()
    # pdb.set_trace()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def parse_det_offset(Y, cfg, nms_algorithm, score=0.1,down=4):
    seman = Y[0][0, 0, :, :]
    height = Y[1][0, 0, :, :]
    offset_y = Y[2][0, 0, :, :]
    offset_x = Y[2][0, 1, :, :]
    y_c, x_c = np.where(seman > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = 0.41*h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, cfg.dataset.size_test[1]), min(y1 + h, cfg.dataset.size_test[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        keep = NMS(boxs, cfg.test_cfg.nms_threshold, nms_algorithm)
        boxs = boxs[keep, :]
    return boxs
