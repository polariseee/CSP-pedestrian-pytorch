import os
import pickle
import numpy as np


class GenCaltech(object):
    def __init__(self):
        self.root_dir = './Caltech/train'
        self.all_img_path = os.path.join(self.root_dir, 'IMG')
        self.all_anno_path = os.path.join(self.root_dir, 'anno_train10x_alignedby_RotatedFilters/')
        self.res_path_gt = './data/cache/caltech/train_gt'
        self.res_path_nogt = './data/cache/caltech/train_nogt'
        self.rows = 480
        self.cols = 640
        self.image_data_gt = []
        self.image_data_nogt = []
        self._gen_cache()
        self._write_pickle()

    def _gen_cache(self):
        valid_count = 0
        iggt_count = 0
        box_count = 0
        files = sorted(os.listdir(self.all_anno_path))
        # ratio = os.path.join(self.root_dir, 'ratio.txt')
        # bbox_area_1 = os.path.join(self.root_dir, 'bbox_area_1.txt')
        # bf = open(ratio, 'w')
        # bf_1 = open(bbox_area_1, 'w')
        for l in range(len(files)):
            gtname = files[l]
            imgname = files[l].split('.')[0] + '.jpg'
            img_path = os.path.join(self.all_img_path, imgname)
            gt_path = os.path.join(self.all_anno_path, gtname)

            boxes = []
            ig_boxes = []
            with open(gt_path, 'rb') as fid:
                lines = fid.readlines()
            if len(lines) > 1:
                for i in range(1, len(lines)):
                    info = lines[i].strip().split(' '.encode())
                    label = info[0]
                    occ, ignore = info[5], info[10]
                    x1, y1 = max(int(float(info[1])), 0), max(int(float(info[2])), 0)
                    w, h = min(int(float(info[3])), self.cols - x1 - 1), min(int(float(info[4])), self.rows - y1 - 1)
                    box = np.array([int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)])
                    if int(ignore) == 0:
                        boxes.append(box)
                        # bf.write(str((box[3] - box[1]) / (box[2] - box[0])))
                        # bf.write('\r\n')
                        # if ((box[2] - box[0]) * (box[3] - box[1])) < 1024 or ((box[2] - box[0]) * (box[3] - box[1])) > 262144:
                        #     bf_1.write(str((box[2] - box[0]) * (box[3] - box[1])))
                        #     bf_1.write('\r\n')
                    else:
                        ig_boxes.append(box)
            boxes = np.array(boxes)
            ig_boxes = np.array(ig_boxes)

            annotation = {}
            annotation['filepath'] = img_path
            box_count += len(boxes)
            iggt_count += len(ig_boxes)
            annotation['bboxes'] = boxes
            annotation['ignoreareas'] = ig_boxes
            if len(boxes) == 0:
                assert len(annotation['bboxes']) == 0
                self.image_data_nogt.append(annotation)
            else:
                assert len(annotation['bboxes']) != 0
                self.image_data_gt.append(annotation)
                valid_count += 1
        print('{} images and {} valid images, {} valid gt and {} ignored gt'.format(len(files), valid_count, box_count,
                                                                                    iggt_count))

    def _write_pickle(self):
        if not os.path.exists(self.res_path_gt):
            with open(self.res_path_gt, 'wb') as fid:
                pickle.dump(self.image_data_gt, fid)
        if not os.path.exists(self.res_path_nogt):
            with open(self.res_path_nogt, 'wb') as fid:
                pickle.dump(self.image_data_nogt, fid)


GenCaltech()
