from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C


__C.DATASET_NAME = 'coco'  # Dataset name: coco, others(TODO)
__C.CONFIG_NAME = 'config_test1'  # 当前配置
__C.DATA_DIR = '/home/zjj/data/coco'
__C.GPU_NUM = 1  # 使用GPU数
__C.WORKERS = 6  # 加载数据的子进程数目


__C.B_VALIDATION = True  # True表示测试时生成所有测试集图片，Fasle表示生成单张图片

# Dataset options
__C.DATASET = edict()
__C.DATASET.MAX_OBJ_NUM = 8
__C.DATASET.MIN_OBJ_NUM = 3
__C.DATASET.IMAGE_SIZE = (256, 256)



# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True  # True为训练模式
__C.TRAIN.BATCH_SIZE = 100
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 25

__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.DISCRIMINATOR1_LR = 2e-4
__C.TRAIN.DISCRIMINATOR2_LR = 2e-4 

__C.TRAIN.NET_G = ''  # 训练好的生成器(鉴别器)路径

__C.TRAIN.LOSS = edict()  # 系数
__C.TRAIN.LOSS.G_LAMBDA_ADVERSIAL = 1.0
__C.TRAIN.LOSS.G_LAMBDA_PER_OBJ = 2.0
__C.TRAIN.LOSS.G_LAMBDA_OTHER = 5.0







def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)
