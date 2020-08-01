# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .cpu_nms import nms, soft_nms
# from .nms import py_cpu_nms
import numpy as np


# def Soft_Nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):
#
#     keep = soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
#                         np.float32(sigma), np.float32(Nt),
#                         np.float32(threshold),
#                         np.uint8(method))
#     return keep


def NMS(dets, thresh, nms_algorithm, soft=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if soft:
        return soft_nms(np.ascontiguousarray(dets, dtype=np.float32), Nt=thresh, method=nms_algorithm)
    else:
        return nms(dets, thresh)
