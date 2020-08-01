#!/usr/bin/env python
import os
import torch
import argparse
import importlib

from nnet.py_factory import NetworkFactory
from mmcv import Config
import pdb

torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args


def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def test(cfg, test_epoch, debug=False):

    max_epochs = cfg.train_cfg.num_epochs

    print("building neural network...")
    nnet = NetworkFactory(cfg)

    test_file = "test.{}".format(cfg.test_cfg.sample_module)
    testing = importlib.import_module(test_file).testing

    nnet.cuda()
    nnet.eval_mode()

    for epoch in range(test_epoch, max_epochs+1):
        result_dir = os.path.join(cfg.test_cfg.save_dir, str(epoch))
        make_dirs([result_dir])
        print("loading parameters at epoch: {}".format(epoch))
        nnet.load_params(epoch)

        testing(cfg, nnet, result_dir, debug=debug)


if __name__ == "__main__":
    args = parse_args()

    print("cfg_file: {}".format(args.cfg_file))

    cfg_file = Config.fromfile(args.cfg_file)
    cfg_file.test_cfg.test = True

    print("system config...")
    print(f'Config:\n{cfg_file.pretty_text}')

    test(cfg_file, args.testiter, args.debug)
