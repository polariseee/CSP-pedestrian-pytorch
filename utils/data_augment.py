import cv2
import numpy as np
import random


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def color_jittering_(data_rng, image):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)


def brightness(image, min=0.5, max=2.0):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_br = np.random.uniform(min, max)
    mask = hsv[:, :, 2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
    hsv[:, :, 2] = v_channel

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def resize_image(image, gts, igs, scale=[0.4, 1.5]):
    height, width = image.shape[0:2]
    ratio = np.random.uniform(scale[0], scale[1])
    # if len(gts)>0 and np.max(gts[:,3]-gts[:,1])>300:
    #     ratio = np.random.uniform(scale[0], 1.0)
    new_height, new_width = int(ratio*height), int(ratio*width)
    image = cv2.resize(image, (new_width, new_height))
    if len(gts) > 0:
        gts = np.asarray(gts,dtype=float)
        gts[:, 0:4:2] *= ratio
        gts[:, 1:4:2] *= ratio

    if len(igs) > 0:
        igs = np.asarray(igs, dtype=float)
        igs[:, 0:4:2] *= ratio
        igs[:, 1:4:2] *= ratio

    return image, gts, igs


def random_crop(image, gts, igs, crop_size, limit=8):
    img_height, img_width = image.shape[0:2]
    crop_h, crop_w = crop_size

    if len(gts)>0:
        sel_id = np.random.randint(0, len(gts))
        sel_center_x = int((gts[sel_id, 0] + gts[sel_id, 2]) / 2.0)
        sel_center_y = int((gts[sel_id, 1] + gts[sel_id, 3]) / 2.0)
    else:
        sel_center_x = int(np.random.randint(0, img_width - crop_w+1) + crop_w * 0.5)
        sel_center_y = int(np.random.randint(0, img_height - crop_h+1) + crop_h * 0.5)

    crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
    crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
    diff_x = max(crop_x1 + crop_w - img_width, int(0))
    crop_x1 -= diff_x
    diff_y = max(crop_y1 + crop_h - img_height, int(0))
    crop_y1 -= diff_y
    cropped_image = np.copy(image[crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w])
    # crop detections
    if len(igs)>0:
        igs[:, 0:4:2] -= crop_x1
        igs[:, 1:4:2] -= crop_y1
        igs[:, 0:4:2] = np.clip(igs[:, 0:4:2], 0, crop_w)
        igs[:, 1:4:2] = np.clip(igs[:, 1:4:2], 0, crop_h)
        keep_inds = ((igs[:, 2] - igs[:, 0]) >=8) & \
                    ((igs[:, 3] - igs[:, 1]) >=8)
        igs = igs[keep_inds]
    if len(gts)>0:
        ori_gts = np.copy(gts)
        gts[:, 0:4:2] -= crop_x1
        gts[:, 1:4:2] -= crop_y1
        gts[:, 0:4:2] = np.clip(gts[:, 0:4:2], 0, crop_w)
        gts[:, 1:4:2] = np.clip(gts[:, 1:4:2], 0, crop_h)

        before_area = (ori_gts[:, 2] - ori_gts[:, 0]) * (ori_gts[:, 3] - ori_gts[:, 1])
        after_area = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])

        keep_inds = ((gts[:, 2] - gts[:, 0]) >=limit) & \
                    (after_area >= 0.5 * before_area)
        gts = gts[keep_inds]

    return cropped_image, gts, igs


def random_pave(image, gts, igs, pave_size, limit=8):
    img_height, img_width = image.shape[0:2]
    pave_h, pave_w = pave_size
    # paved_image = np.zeros((pave_h, pave_w, 3), dtype=image.dtype)
    paved_image = np.ones((pave_h, pave_w, 3), dtype=image.dtype)*np.mean(image,dtype=int)
    pave_x = int(np.random.randint(0, pave_w-img_width+1))
    pave_y = int(np.random.randint(0, pave_h-img_height+1))
    paved_image[pave_y:pave_y+img_height, pave_x:pave_x+img_width] = image
    # pave detections
    if len(igs) > 0:
        igs[:, 0:4:2] += pave_x
        igs[:, 1:4:2] += pave_y
        keep_inds = ((igs[:, 2] - igs[:, 0]) >=8) & \
                    ((igs[:, 3] - igs[:, 1]) >=8)
        igs = igs[keep_inds]

    if len(gts) > 0:
        gts[:, 0:4:2] += pave_x
        gts[:, 1:4:2] += pave_y
        keep_inds = ((gts[:, 2] - gts[:, 0]) >= limit)
        gts = gts[keep_inds]

    return paved_image, gts, igs
