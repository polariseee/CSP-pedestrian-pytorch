import os
import cv2
import numpy as np
import torch
import time

from tqdm import tqdm
import pdb


def kp_decode(nnet, images, **kwargs):
    detections = nnet.test([images], **kwargs)
    return detections


def kp_detection(cfg, nnet, result_dir, debug=False, decode_func=kp_decode):
    if debug:
        debug_dir = os.path.join(result_dir, "debug")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

    img_txt_file = os.path.join(cfg.test_cfg.data_dir, 'test.txt')
    with open(img_txt_file, 'r') as fid:
        filenames = fid.readlines()
    img_name_list = []
    for filename in filenames:
        filename = filename.strip('\n') + '.jpg'
        img_name_list.append(filename)

    num_images = len(img_name_list)

    img_dir = os.path.join(cfg.test_cfg.data_dir, 'IMG')

    all_time = []

    for st in range(6, 11):
        set_path = os.path.join(result_dir, 'set' + '%02d' % st)
        if not os.path.exists(set_path):
            os.mkdir(set_path)

    for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        img_name = img_name_list[ind]

        img_meta = {}
        filepath = os.path.join(img_dir, img_name)
        filepath_next = os.path.join(img_dir, img_name_list[ind + 1]) if ind < num_images - 1 else \
            os.path.join(img_dir, img_name_list[ind])
        set = filepath.split('/')[-1].split('_')[0]
        video = filepath.split('/')[-1].split('_')[1]
        frame_number = int(filepath.split('/')[-1].split('_')[2][1:6]) + 1
        frame_number_next = int(filepath_next.split('/')[-1].split('_')[2][1:6]) + 1
        set_path = os.path.join(result_dir, set)
        video_path = os.path.join(set_path, video + '.txt')

        if frame_number == 30:
            res_all = []

        t1 = time.time()
        image = cv2.imread(filepath)
        img_meta['filename'] = filepath
        img_meta['img_shape'] = image.shape
        img_meta['scale_factor'] = np.array([1., 1., 1., 1.], dtype=np.float32)
        image = image.astype(np.float32)
        image[:, :, 0] -= cfg.dataset.img_channel_mean[0]
        image[:, :, 1] -= cfg.dataset.img_channel_mean[1]
        image[:, :, 2] -= cfg.dataset.img_channel_mean[2]

        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)

        detections = decode_func(nnet, image, **img_meta)
        t2 = time.time()
        t = t2 - t1
        all_time.append(round(t, 3))

        if debug:
            img = img_name_list[ind]

            imgpath = os.path.join(img_dir, img)
            image = cv2.imread(imgpath)

            cat_name = 'person'
            cat_size = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            color = np.random.random((3,)) * 0.6 + 0.4
            color = color * 255
            color = color.astype(np.int32).tolist()
            for bbox in detections:
                bbox = bbox[0:4].astype(np.int32)
                if bbox[1] - cat_size[1] - 2 < 0:
                    cv2.rectangle(image,
                                  (bbox[0], bbox[1] + 2),
                                  (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                                  color, -1
                                  )
                    cv2.putText(image, cat_name,
                                (bbox[0], bbox[1] + cat_size[1] + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                                )
                else:
                    cv2.rectangle(image,
                                  (bbox[0], bbox[1] - cat_size[1] - 2),
                                  (bbox[0] + cat_size[0], bbox[1] - 2),
                                  color, -1
                                  )
                    cv2.putText(image, cat_name,
                                (bbox[0], bbox[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                                )
                cv2.rectangle(image,
                              (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]),
                              color, 2
                              )
            debug_file = os.path.join(debug_dir, img)
            cv2.imwrite(debug_file, image)
        else:
            if len(detections) > 0:
                f_res = np.repeat(frame_number, len(detections), axis=0).reshape((-1, 1))
                detections[:, [2, 3]] -= detections[:, [0, 1]]
                res_all += np.concatenate((f_res, detections), axis=-1).tolist()
            if frame_number_next == 30 or ind == num_images - 1:
                np.savetxt(video_path, np.array(res_all), fmt='%6f')
    print('Inference time used: %.3f' % (sum(all_time) / num_images))
    return 0


def testing(cfg, nnet, result_dir, debug=False):
    return globals()[cfg.dataset.sampling_function](cfg, nnet, result_dir, debug=debug)
