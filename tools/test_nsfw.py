#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-2-20 上午10:48
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_nsfw.py.py
# @IDE: PyCharm
"""
Test nsfw model script
"""
import os.path as ops
import argparse
import glog as log

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from config import global_config
from nsfw_model import nsfw_classification_net
from data_provider import nsfw_data_feed_pipline


CFG = global_config.cfg

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default=None, help='The dataset dir')
    parser.add_argument('--image_path', type=str, default=None, help='The image path')
    parser.add_argument('--weights_path', type=str, help='The model weights file path')

    return parser.parse_args()


def scale_image(image):
    """

    :param image:
    :return:
    """
    means = tf.expand_dims(tf.expand_dims(_CHANNEL_MEANS, 0), 0)

    return image + means


def calculate_top_k_error(predictions, labels, k=1):
    """
    Calculate the top-k error
    :param predictions: 2D tensor with shape [batch_size, num_labels]
    :param labels: 1D tensor with shape [batch_size, 1]
    :param k: int
    :return: tensor with shape [1]
    """
    batch_size = CFG.TEST.BATCH_SIZE
    in_top_k = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
    num_correct = tf.reduce_sum(in_top_k)

    return (batch_size - num_correct) / float(batch_size)


def nsfw_eval_dataset(dataset_dir, weights_path, top_k=1):
    """
    Evaluate the nsfw dataset
    :param dataset_dir: The nsfw dataset dir which contains tensorflow records file
    :param weights_path: The pretrained nsfw model weights file path
    :param top_k: calculate the top k accuracy
    :return:
    """
    assert ops.exists(dataset_dir)

    # set nsfw data feed pipline
    test_dataset = nsfw_data_feed_pipline.NsfwDataFeeder(dataset_dir=dataset_dir,
                                                         flags='test')
    prediciton_map = test_dataset.prediction_map

    with tf.device('/gpu:1'):
        # set nsfw classification model
        phase = tf.constant('test', dtype=tf.string)

        # set nsfw net
        nsfw_net = nsfw_classification_net.NSFWNet(phase=phase,
                                                   resnet_size=CFG.NET.RESNET_SIZE)

        # compute train loss
        images, labels = test_dataset.inputs(batch_size=CFG.TEST.BATCH_SIZE,
                                             num_epochs=1)

        images_scale = tf.map_fn(fn=scale_image, elems=images, dtype=tf.float32)

        logits = nsfw_net.inference(input_tensor=images,
                                    name='nsfw_cls_model',
                                    reuse=False)

        predictions = tf.nn.softmax(logits)
        top_k_error = calculate_top_k_error(predictions, labels, top_k)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        CFG.TRAIN.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    # set tensorflow saver
    saver = tf.train.Saver(variables_to_restore)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    avg_prediction_top_k_accuracy = []

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        while True:
            try:
                images_vals, predictions_vals, labels_vals, top1_error_val = sess.run(
                    fetches=[images_scale,
                             predictions,
                             labels,
                             top_k_error])

                log.info('\n**************')
                log.info('Test dataset batch size: {:d}'.format(images_vals.shape[0]))
                log.info('---- Sample Id ---- Gt label ---- Prediction ----')

                for index, image in enumerate(images_vals):

                    label_gt = prediciton_map[labels_vals[index]]

                    prediction_score = dict()

                    for score_index, score in enumerate(predictions_vals[index]):
                        prediction_score[prediciton_map[score_index]] = format(score, '.5f')

                    log.info('---- {:d} ---- {:s} ---- {}'.format(index, label_gt, prediction_score))

                    # plt.ion()
                    # plt.figure('source image')
                    # plt.imshow(np.array(image[:, :, (2, 1, 0)], np.uint8))
                    # plt.pause(5.0)
                    # plt.show()
                    # plt.ioff()

                log.info('Top 1 accuracy of this batch is: {:.5f}'.format(1 - top1_error_val))
                avg_prediction_top_k_accuracy.append(1 - top1_error_val)

            except tf.errors.OutOfRangeError as err:
                log.info('Total avg top 1 accuracy is: {:.5f}'.format(np.mean(avg_prediction_top_k_accuracy)))
                break
    return


def nsfw_classify_image(image_path, weights_path):
    """
    Use nsfw model to classify a single image
    :param image_path: The image file path
    :param weights_path: The pretrained weights file path
    :return:
    """
    assert ops.exists(image_path)

    prediciton_map = global_config.NSFW_PREDICT_MAP

    with tf.device('/gpu:1'):
        # set nsfw classification model

        image_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[1, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3],
                                      name='input_tensor')
        # set nsfw net
        phase = tf.constant('test', dtype=tf.string)
        nsfw_net = nsfw_classification_net.NSFWNet(phase=phase,
                                                   resnet_size=CFG.NET.RESNET_SIZE)

        # compute inference logits
        logits = nsfw_net.inference(input_tensor=image_tensor,
                                    name='nsfw_cls_model',
                                    reuse=False)

        predictions = tf.nn.softmax(logits)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        CFG.TRAIN.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    # set tensorflow saver
    saver = tf.train.Saver(variables_to_restore)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_vis = image
        image = cv2.resize(src=image,
                           dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                           interpolation=cv2.INTER_CUBIC)
        image = np.array(image, dtype=np.float32) - np.array(_CHANNEL_MEANS, np.float32)

        predictions_vals = sess.run(
            fetches=predictions,
            feed_dict={image_tensor: [image]})

        prediction_score = dict()

        for score_index, score in enumerate(predictions_vals[0]):
            prediction_score[prediciton_map[score_index]] = format(score, '.5f')

        log.info('Predict result is: {}'.format(prediction_score))

        plt.figure('source image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.show()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # test net
    if args.dataset_dir is not None and args.image_path is None:
        nsfw_eval_dataset(args.dataset_dir, args.weights_path)
    elif args.dataset_dir is None and args.image_path is not None:
        nsfw_classify_image(args.image_path, args.weights_path)
    else:
        raise ValueError('You can not test a single image and test the nsfw dataset at the meantime')
