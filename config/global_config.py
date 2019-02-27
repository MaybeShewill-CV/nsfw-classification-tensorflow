#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-31 上午11:21
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : global_config.py
# @IDE: PyCharm Community Edition
"""
设置全局变量
"""
import easydict

__C = easydict.EasyDict()
# Consumers can get config by: from config import cfg

cfg = __C

# Train options
__C.TRAIN = easydict.EasyDict()

# Set the shadownet training epochs
__C.TRAIN.EPOCHS = 100010
# Set the display step
__C.TRAIN.DISPLAY_STEP = 1
# Set the test display step during training process
__C.TRAIN.VAL_DISPLAY_STEP = 1000
# Set the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.1
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.75
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = False
# Set the shadownet training batch size
__C.TRAIN.BATCH_SIZE = 32
# Set the shadownet validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 32
# Set the learning rate decay steps
__C.TRAIN.LR_DECAY_STEPS_1 = 40000
# Set the learning rate decay steps
__C.TRAIN.LR_DECAY_STEPS_2 = 80000
# Set the learning rate decay rate
__C.TRAIN.LR_DECAY_RATE = 0.1
# Set the weights decay
__C.TRAIN.WEIGHT_DECAY = 0.0001
# Set the train moving average decay
__C.TRAIN.MOVING_AVERAGE_DECAY = 0.9999
# Set the class numbers
__C.TRAIN.CLASSES_NUMS = 5
# Set the image height
__C.TRAIN.IMG_HEIGHT = 256
# Set the image width
__C.TRAIN.IMG_WIDTH = 256
# Set the image height
__C.TRAIN.CROP_IMG_HEIGHT = 224
# Set the image width
__C.TRAIN.CROP_IMG_WIDTH = 224
# Set the GPU nums
__C.TRAIN.GPU_NUM = 2
# Set cpu multi process thread nums
__C.TRAIN.CPU_MULTI_PROCESS_NUMS = 6

# Test options
__C.TEST = easydict.EasyDict()

# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.8
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = True
# Set the test batch size
__C.TEST.BATCH_SIZE = 64

__C.NET = easydict.EasyDict()
# Set net residual_blocks_nums
__C.NET.RESNET_SIZE = 50
# Set feats summary flag
__C.NET.NEED_SUMMARY_FEATS_MAP = False

# Set nsfw dataset label map
NSFW_LABEL_MAP = {'drawing': 0, 'hentai': 1, 'neural': 2, 'porn': 3, 'sexy': 4}
# Set nsfw dataset prediction map
NSFW_PREDICT_MAP = {0: 'drawing', 1: 'hentai', 2: 'neural', 3: 'porn', 4: 'sexy'}
