#!/usr/bin/env python
import os

import torch
import numpy as np
import argparse
import importlib
import traceback
import pickle
import json
import time
import random

from mmcv import Config
from mmcv.utils import get_logger
from nnet import NetworkFactory
from torch.multiprocessing import Process, Queue
import pdb
torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train CSP")
    parser.add_argument("config", help="train config file path", type=str)
    parser.add_argument("--work-dir", help='the dir to save logs and models')
    parser.add_argument("--iter", dest="start_epoch",
                        help="train at epoch i",
                        default=0, type=int)

    args = parser.parse_args()
    return args


def prefetch_data(cfg, queue, sample_data, ped_data=None, emp_data=None):
    ind = 0
    n_ind = 0
    random.shuffle(ped_data)
    random.shuffle(emp_data)
    print("start prefetching data...")
    while True:
        try:
            data, ind, n_ind = sample_data(cfg.dataset, ped_data, ind, emp_data, n_ind)  # ind可以保证每个batch取得数据不相同
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e


def init_parallel_jobs(cfg, queue, fn, ped_data=None, emp_data=None):
    tasks = Process(target=prefetch_data, args=(cfg, queue, fn, ped_data, emp_data))
    tasks.daemon = True
    tasks.start()
    return tasks


def train(logger, json_file, cfg, start_epoch=0):
    learning_rate    = cfg.train_cfg.learning_rate
    pretrained_model = cfg.train_cfg.pretrain
    display          = cfg.train_cfg.display
    sample_module    = cfg.train_cfg.sample_module
    iter_per_epoch   = cfg.train_cfg.iter_per_epoch
    num_epochs       = cfg.train_cfg.num_epochs
    batch_size       = cfg.dataset.batch_size

    # queues storing data for training
    training_queue   = Queue(cfg.train_cfg.prefetch_size)

    # load data sampling function
    data_file   = "sample.{}".format(sample_module)
    sample_data = importlib.import_module(data_file).sample_data

    if cfg.train_cfg.cache_ped:
        with open(cfg.train_cfg.cache_ped, 'rb') as fid:
            ped_data = pickle.load(fid)
    if cfg.train_cfg.cache_emp:
        with open(cfg.train_cfg.cache_emp, 'rb') as fid:
            emp_data = pickle.load(fid)
    length_dataset = len(ped_data)+len(emp_data)
    logger.info('the length of dataset is: {}'.format(length_dataset))

    # allocating resources for parallel reading
    if cfg.train_cfg.cache_emp:
        training_tasks   = init_parallel_jobs(cfg, training_queue, sample_data, ped_data, emp_data)
    else:
        training_tasks = init_parallel_jobs(cfg, training_queue, sample_data, ped_data)
    # prefetch_data(cfg, training_queue, sample_data, ped_data, emp_data)

    logger.info("building model...")

    nnet = NetworkFactory(cfg)

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        logger.info("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_epoch:
        nnet.load_params(start_epoch)
        nnet.set_lr(learning_rate)
        logger.info("training starts from iteration {} with learning_rate {}".format(start_epoch, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    logger.info("training start...")
    nnet.cuda()
    nnet.train_mode()
    epoch_length = int(iter_per_epoch / batch_size)
    json_obj = open(json_file, 'w')
    loss = []
    for epoch in range(start_epoch, num_epochs):
        for iteration in range(1, epoch_length + 1):
            training = training_queue.get(block=True)
            training_loss = nnet.train(**training)

            loss.append(training_loss.item())

            if display and iteration % display == 0:
                loss = np.array(loss)
                logger.info("Epoch: {}/{}, loss_csp: {:.5f}".format(epoch+1, num_epochs, loss.sum() / display))
                text = {"Epoch": epoch+1, "loss_csp": round(loss.sum() / display, 5)}
                text = json.dumps(text)
                json_obj.write(text)
                json_obj.write('\r\n')
                loss = []

            del training_loss

        nnet.save_params(epoch + 1)

    # terminating data fetching processes
    training_tasks.terminate()


if __name__ == "__main__":
    args = parse_args()

    cfg_file = Config.fromfile(args.config)

    if args.work_dir:
        cfg_file.train_cfg.work_dir = args.work_dir

    if not os.path.exists(cfg_file.train_cfg.work_dir):
        os.makedirs(cfg_file.train_cfg.work_dir)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg_file.train_cfg.work_dir, f'{timestamp}.log')
    logger = get_logger(name='CAP', log_file=log_file)

    json_file = os.path.join(cfg_file.train_cfg.work_dir, f'{timestamp}.json')

    logger.info("system config...")
    logger.info(f'Config:\n{cfg_file.pretty_text}')

    train(logger, json_file, cfg_file, args.start_epoch)
