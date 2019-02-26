#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-2-15 下午2:13
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : tf_io_pipline_tools.py
# @IDE: PyCharm
"""
Some tensorflow records io tools
"""
import os
import os.path as ops

import cv2
import tensorflow as tf
import glog as log

from config import global_config


CFG = global_config.cfg

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]


def int64_feature(value):
    """

    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """

    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_example_tfrecords(example_paths, example_labels, tfrecords_path):
    """
    write tfrecords
    :param example_paths:
    :param example_labels:
    :param tfrecords_path:
    :return:
    """
    _tfrecords_dir = ops.split(tfrecords_path)[0]
    os.makedirs(_tfrecords_dir, exist_ok=True)

    log.info('Writing {:s}....'.format(tfrecords_path))

    with tf.python_io.TFRecordWriter(tfrecords_path) as _writer:
        for _index, _example_path in enumerate(example_paths):

            with open(_example_path, 'rb') as f:
                check_chars = f.read()[-2:]
            if check_chars != b'\xff\xd9':
                log.error('Image file {:s} is not complete'.format(_example_path))
                continue
            else:
                _example_image = cv2.imread(_example_path, cv2.IMREAD_COLOR)
                _example_image = cv2.resize(_example_image,
                                            dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                            interpolation=cv2.INTER_CUBIC)
                _example_image_raw = _example_image.tostring()

                _example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'height': int64_feature(CFG.TRAIN.IMG_HEIGHT),
                            'width': int64_feature(CFG.TRAIN.IMG_WIDTH),
                            'depth': int64_feature(3),
                            'label': int64_feature(example_labels[_index]),
                            'image_raw': bytes_feature(_example_image_raw)
                        }))
                _writer.write(_example.SerializeToString())

    log.info('Writing {:s} complete'.format(tfrecords_path))

    return


def decode(serialized_example):
    """
    Parses an image and label from the given `serialized_example`
    :param serialized_example:
    :return:
    """
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64)
        })

    # decode image
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image_shape = tf.stack([CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3])
    image = tf.reshape(image, image_shape)

    # Convert label from a scalar int64 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def augment(image, label):
    """

    :param image:
    :param label:
    :return:
    """
    # TODO luoyao(luoyao@baidu.com) Here can apply some augmentation functions

    return image, label


def normalize(image, label):
    """
    Normalize the image data by substracting the imagenet mean value
    :param image:
    :param label:
    :return:
    """

    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    means = tf.expand_dims(tf.expand_dims(_CHANNEL_MEANS, 0), 0)

    return image - means, label
