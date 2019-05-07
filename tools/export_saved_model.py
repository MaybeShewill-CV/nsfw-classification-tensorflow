#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-2-22 上午11:10
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : export_saved_model.py
# @IDE: PyCharm
"""
Build tensorflow saved model for tensorflowjs converter to use
"""
import os.path as ops
import argparse
import glog as log

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import saved_model as sm

from config import global_config
from nsfw_model import nsfw_classification_net

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
    parser.add_argument('--export_dir', type=str, help='The model export dir')
    parser.add_argument('--ckpt_path', type=str, help='The pretrained ckpt model weights file path')

    return parser.parse_args()


def central_crop(image, central_fraction):
    """

    :param image:
    :param central_fraction:
    :return:
    """

    image_shape = np.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    if central_fraction >= 1 or central_fraction < 0:
        raise ValueError('Central fraction should be in [0, 1)')

    top = int((1 - central_fraction) * image_height / 2)
    bottom = image_height - top
    left = int((1 - central_fraction) * image_width / 2)
    right = image_width - left

    if not image_height or not image_width:
        raise ValueError('Image shape with zero')

    if len(image_shape) == 2:
        crop_image = image[top:bottom, left:right]
        return crop_image
    elif len(image_shape) == 3:
        crop_image = image[top:bottom, left:right, :]
        return crop_image
    else:
        raise ValueError('Wrong image shape')


def build_saved_model(ckpt_path, export_dir):
    """
    Convert source ckpt weights file into tensorflow saved model
    :param ckpt_path:
    :param export_dir:
    :return:
    """

    if ops.exists(export_dir):
        raise ValueError('Export dir must be a dir path that does not exist')

    assert ops.exists(ops.split(ckpt_path)[0])

    # build inference tensorflow graph
    image_tensor = tf.placeholder(dtype=tf.float32,
                                  shape=[1, CFG.TRAIN.CROP_IMG_HEIGHT, CFG.TRAIN.CROP_IMG_WIDTH, 3],
                                  name='input_tensor')
    # set nsfw net
    phase = tf.constant('test', dtype=tf.string)
    nsfw_net = nsfw_classification_net.NSFWNet(phase=phase,
                                               resnet_size=CFG.NET.RESNET_SIZE)

    # compute inference logits
    logits = nsfw_net.inference(input_tensor=image_tensor,
                                name='nsfw_cls_model',
                                reuse=False)

    predictions = tf.nn.softmax(logits, name='nsfw_cls_model/final_prediction')

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

        saver.restore(sess=sess, save_path=ckpt_path)

        # set model save builder
        saved_builder = sm.builder.SavedModelBuilder(export_dir)

        # add tensor need to be saved
        saved_input_tensor = sm.utils.build_tensor_info(image_tensor)
        saved_prediction_tensor = sm.utils.build_tensor_info(predictions)

        # build SignatureDef protobuf
        signatur_def = sm.signature_def_utils.build_signature_def(
            inputs={'input_tensor': saved_input_tensor},
            outputs={'prediction': saved_prediction_tensor},
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
        )

        # add graph into MetaGraphDef protobuf
        saved_builder.add_meta_graph_and_variables(
            sess,
            tags=[sm.tag_constants.SERVING],
            signature_def_map={'classify_result': signatur_def}
        )

        # save model
        saved_builder.save()

    return


def test_load_saved_model(saved_model_dir):
    """

    :param saved_model_dir:
    :return:
    """

    prediciton_map = global_config.NSFW_PREDICT_MAP

    image = cv2.imread('data/test_data/drawing_16715.jpg', cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(src=image,
                       dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                       interpolation=cv2.INTER_CUBIC)
    image = central_crop(image=image,
                         central_fraction=CFG.TRAIN.CROP_IMG_HEIGHT / CFG.TRAIN.IMG_HEIGHT)
    image = np.array(image, dtype=np.float32) - np.array(_CHANNEL_MEANS, np.float32)
    image = np.expand_dims(image, 0)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        meta_graphdef = sm.loader.load(
            sess,
            tags=[sm.tag_constants.SERVING],
            export_dir=saved_model_dir)

        signature_def_d = meta_graphdef.signature_def
        signature_def_d = signature_def_d['classify_result']

        image_input_tensor = signature_def_d.inputs['input_tensor']
        prediction_tensor = signature_def_d.outputs['prediction']

        input_tensor = sm.utils.get_tensor_from_tensor_info(image_input_tensor, sess.graph)
        predictions = sm.utils.get_tensor_from_tensor_info(prediction_tensor, sess.graph)

        prediction_val = sess.run(predictions, feed_dict={input_tensor: image})

        prediction_score = dict()

        for score_index, score in enumerate(prediction_val[0]):
            prediction_score[prediciton_map[score_index]] = format(score, '.5f')

        log.info('Predict result: {}'.format(prediction_score))

        plt.figure('source image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])


if __name__ == '__main__':
    """
    build saved model
    """
    # init args
    args = init_args()

    # build saved model
    build_saved_model(args.ckpt_path, args.export_dir)

    # test build saved model
    test_load_saved_model(args.export_dir)
