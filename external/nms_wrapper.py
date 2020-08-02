# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .nms.cpu_nms import nms, soft_nms
from .nms.gpu_nms import gpu_nms
import numpy as np


# def Soft_Nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):
#
#     keep = soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
#                         np.float32(sigma), np.float32(Nt),
#                         np.float32(threshold),
#                         np.uint8(method))
#     return keep


def NMS(dets, thresh, nms_algorithm, usegpu=False, gpu_id=0):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if nms_algorithm == 1 or nms_algorithm == 2:
        return soft_nms(np.ascontiguousarray(dets, dtype=np.float32), Nt=thresh, method=nms_algorithm)
    else:
        if usegpu:
            return gpu_nms(dets, thresh, device_id=gpu_id)
        else:
            return nms(dets, thresh)
