import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

from dataset import CocoDataset, get_dataloader, batch_mask_image
from trainer import CondGANTrainer
from miscc.config import cfg, cfg_from_file

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

np.seterr(divide='ignore', invalid='ignore')  #消除向量中除以0的警告


# 限制使用的GPU个数
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def parse_args():
    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='/home/zjj/data/coco')
    parser.add_argument('--manualSeed', type=int, help='manual seed', default=2020)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    print("loading dataloader")
    train_dataloader, val_dataloader = get_dataloader(batch_size=cfg.TRAIN.BATCH_SIZE)
    assert train_dataloader, val_dataloader

    # Define models and go to train/evaluate
    algo = CondGANTrainer(output_dir=output_dir, data_loader=train_dataloader)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        pass
    end_t = time.time()
    print('Total time for training:', end_t - start_t)