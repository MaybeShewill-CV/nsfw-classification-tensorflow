#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-2-15 下午8:53
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : train_nsfw.py
# @IDE: PyCharm
"""
Train nsfw model script
"""
import argparse
import os
import os.path as ops
import time
import math

import numpy as np
import tensorflow as tf
import glog as log

from config import global_config
from data_provider import nsfw_data_feed_pipline
from nsfw_model import nsfw_classification_net


CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, help='The dataset_dir')
    parser.add_argument('--weights_path', type=str, default=None, help='The pretrained weights path')

    return parser.parse_args()


def top_k_error(predictions, labels, k=1):
    """
    Calculate the top-k error
    :param predictions: 2D tensor with shape [batch_size, num_labels]
    :param labels: 1D tensor with shape [batch_size, 1]
    :param k: int
    :return: tensor with shape [1]
    """
    batch_size = predictions.get_shape().as_list()[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
    num_correct = tf.reduce_sum(in_top1)

    return (batch_size - num_correct) / float(batch_size)


def train_net(dataset_dir, weights_path=None):
    """

    :param dataset_dir:
    :param weights_path:
    :return:
    """

    # set nsfw data feed pipline
    train_dataset = nsfw_data_feed_pipline.NsfwDataFeeder(dataset_dir=dataset_dir,
                                                          flags='train')
    val_dataset = nsfw_data_feed_pipline.NsfwDataFeeder(dataset_dir=dataset_dir,
                                                        flags='val')

    with tf.device('/gpu:1'):
        # set nsfw classification model
        phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')
        example_tensor_shape = tf.stack([CFG.TRAIN.BATCH_SIZE,
                                         CFG.TRAIN.IMG_HEIGHT,
                                         CFG.TRAIN.IMG_WIDTH,
                                         3])
        # set nsfw net
        nsfw_net = nsfw_classification_net.NSFWNet(phase=phase)

        # compute train loss
        train_images, train_labels = train_dataset.inputs(batch_size=CFG.TRAIN.BATCH_SIZE,
                                                          num_epochs=1)
        train_images = tf.reshape(train_images, example_tensor_shape)
        train_loss = nsfw_net.compute_loss(input_tensor=train_images,
                                           labels=train_labels,
                                           residual_blocks_nums=CFG.NET.RES_BLOCKS_NUMS,
                                           name='nsfw_cls_model',
                                           reuse=False)

        train_logits = nsfw_net.inference(input_tensor=train_images,
                                          residual_blocks_nums=CFG.NET.RES_BLOCKS_NUMS,
                                          name='nsfw_cls_model',
                                          reuse=True)

        train_predictions = tf.nn.softmax(train_logits)
        train_top1_error = top_k_error(train_predictions, train_labels, 1)

        # compute val loss
        val_images, val_labels = val_dataset.inputs(batch_size=CFG.TRAIN.VAL_BATCH_SIZE,
                                                    num_epochs=1)
        val_images = tf.reshape(val_images, example_tensor_shape)
        val_loss = nsfw_net.compute_loss(input_tensor=val_images,
                                         labels=val_labels,
                                         residual_blocks_nums=CFG.NET.RES_BLOCKS_NUMS,
                                         name='nsfw_cls_model',
                                         reuse=True)

        val_logits = nsfw_net.inference(input_tensor=val_images,
                                        residual_blocks_nums=CFG.NET.RES_BLOCKS_NUMS,
                                        name='nsfw_cls_model',
                                        reuse=True)

        val_predictions = tf.nn.softmax(val_logits)
        val_top1_error = top_k_error(val_predictions, val_labels, 1)

    # set tensorflow summary
    tboard_save_path = 'tboard/nsfw_cls'
    os.makedirs(tboard_save_path, exist_ok=True)

    train_loss_scalar = tf.summary.scalar(name='train_loss',
                                          tensor=train_loss)
    train_top1_err_scalar = tf.summary.scalar(name='train_top1_error',
                                              tensor=train_top1_error)
    val_loss_scalar = tf.summary.scalar(name='val_loss',
                                        tensor=val_loss)
    val_top1_err_scalar = tf.summary.scalar(name='val_top1_error',
                                            tensor=val_top1_error)

    train_merge_summary_op = tf.summary.merge([train_loss_scalar, train_top1_err_scalar])

    val_merge_summary_op = tf.summary.merge([val_loss_scalar, val_top1_err_scalar])

    # Set tf saver
    saver = tf.train.Saver()
    model_save_dir = 'model/nsfw_cls'
    os.makedirs(model_save_dir, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'nsfw_cls_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # set optimizer
    with tf.device('/gpu:1'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.polynomial_decay(
            learning_rate=CFG.TRAIN.LEARNING_RATE,
            global_step=global_step,
            decay_steps=CFG.TRAIN.EPOCHS,
            power=0.9)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9).minimize(
                loss=train_loss,
                var_list=tf.trainable_variables(),
                global_step=global_step)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    log.info('Global configuration is as follows:')
    log.info(CFG)

    with sess.as_default():

        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/nsfw_cls_model.pb'.format(model_save_dir))

        if weights_path is None:
            log.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        train_cost_time_mean = []
        val_cost_time_mean = []

        for epoch in range(train_epochs):

            # training part
            t_start = time.time()

            phase_train = 'train'

            _, train_loss_value, train_top1_err_value, train_summary, lr = \
                sess.run(fetches=[optimizer,
                                  train_loss,
                                  train_top1_error,
                                  train_merge_summary_op,
                                  learning_rate],
                         feed_dict={phase: phase_train})

            if math.isnan(train_loss_value):
                log.error('Train loss is nan')
                return

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)

            summary_writer.add_summary(summary=train_summary,
                                       global_step=epoch)

            # validation part
            t_start_val = time.time()

            phase_val = 'test'

            val_loss_value, val_top1_err_value, val_summary = \
                sess.run([val_loss, val_top1_error, val_merge_summary_op],
                         feed_dict={phase: phase_val})

            summary_writer.add_summary(val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                log.info('Epoch_Train: {:d} total_loss= {:6f} top1_error= {:6f} '
                         'lr= {:6f} mean_cost_time= {:5f}s '.
                         format(epoch + 1,
                                train_loss_value,
                                train_top1_err_value,
                                lr,
                                np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            if epoch % CFG.TRAIN.VAL_DISPLAY_STEP == 0:
                log.info('Epoch_Val: {:d} total_loss= {:6f} top1_error= {:6f}'
                         ' mean_cost_time= {:5f}s '.
                         format(epoch + 1,
                                val_loss_value,
                                val_top1_err_value,
                                np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.dataset_dir, args.weights_path)
