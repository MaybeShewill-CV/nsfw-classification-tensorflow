#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-2-28 上午11:20
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/MaybeShewill-CV.github.io
# @File    : evaluate_nsfw.py
# @IDE: PyCharm
"""
Evaluate nsfw model
"""
import itertools
import os.path as ops
import argparse
import glog as log

import tensorflow as tf
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             precision_recall_curve, average_precision_score, f1_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils.fixes import signature
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
    parser.add_argument('--weights_path', type=str, help='The model weights file path')
    parser.add_argument('--top_k', type=int, default=1, help='Evaluate top k error')

    return parser.parse_args()


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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        log.info("Normalized confusion matrix")
    else:
        log.info('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_precision_recall_curve(labels, predictions_prob, class_nums, average_function='weighted'):
    """
    Plot precision recall curve
    :param labels:
    :param predictions_prob:
    :param class_nums:
    :param average_function:
    :return:
    """
    labels = label_binarize(labels, classes=np.linspace(0, class_nums - 1, num=class_nums).tolist())
    predictions_prob = np.array(predictions_prob, dtype=np.float32)

    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(class_nums):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i],
                                                            predictions_prob[:, i])
        average_precision[i] = average_precision_score(labels[:, i], predictions_prob[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision[average_function], recall[average_function], _ = precision_recall_curve(
        labels.ravel(), predictions_prob.ravel())
    average_precision[average_function] = average_precision_score(
        labels, predictions_prob, average=average_function)
    log.info('Average precision score, {:s}-averaged '
             'over all classes: {:.5f}'.format(average_function, average_precision[average_function]))

    plt.figure()
    plt.step(recall[average_function], precision[average_function], color='b', alpha=0.2,
             where='post')
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.fill_between(recall[average_function], precision[average_function], alpha=0.2, color='b',
                     **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, {:s}-averaged over '
        'all classes: AP={:.5f}'.format(average_function, average_precision[average_function]))


def calculate_evaluate_statics(labels, predictions, model_name='Nsfw', avgerage_method='weighted'):
    """
    Calculate Precision, Recall and F1 score
    :param labels:
    :param predictions:
    :param model_name:
    :param avgerage_method:
    :return:
    """
    log.info('Model name: {:s}:'.format(model_name))
    log.info('\tPrecision: {:.5f}'.format(precision_score(y_true=labels,
                                                          y_pred=predictions,
                                                          average=avgerage_method)))
    log.info('\tRecall: {:.5f}'.format(recall_score(y_true=labels,
                                                    y_pred=predictions,
                                                    average=avgerage_method)))
    log.info('\tF1: {:.5f}\n'.format(f1_score(y_true=labels,
                                              y_pred=predictions,
                                              average=avgerage_method)))


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
    class_names = ['drawing', 'hentai', 'neural', 'porn', 'sexy']

    with tf.device('/gpu:1'):
        # set nsfw classification model
        phase = tf.constant('test', dtype=tf.string)

        # set nsfw net
        nsfw_net = nsfw_classification_net.NSFWNet(phase=phase,
                                                   resnet_size=CFG.NET.RESNET_SIZE)

        # compute train loss
        images, labels = test_dataset.inputs(batch_size=CFG.TEST.BATCH_SIZE,
                                             num_epochs=1)

        logits = nsfw_net.inference(input_tensor=images,
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
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # labels overall test dataset
    labels_total = []
    # prediction result overall test dataset
    predictions_total = []
    # prediction score overall test dataset of all subclass
    predictions_prob_total = []

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        while True:
            try:
                predictions_vals, labels_vals = sess.run(
                    fetches=[predictions,
                             labels])

                log.info('**************')
                log.info('Test dataset batch size: {:d}'.format(predictions_vals.shape[0]))
                log.info('---- Sample Id ---- Gt label ---- Prediction ----')

                for index, predictions_val in enumerate(predictions_vals):

                    label_gt = prediciton_map[labels_vals[index]]

                    prediction_score = dict()

                    for score_index, score in enumerate(predictions_val):
                        prediction_score[prediciton_map[score_index]] = format(score, '.5f')

                    log.info('---- {:d} ---- {:s} ---- {}'.format(index, label_gt, prediction_score))

                    # record predicts prob map
                    predictions_prob_total.append(predictions_val.tolist())

                # record total label and prediction results
                labels_total.extend(labels_vals.tolist())
                predictions_total.extend(np.argmax(predictions_vals, axis=1).tolist())

            except tf.errors.OutOfRangeError as err:
                log.info('Loop overall the test dataset')
                break
            except Exception as err:
                log.error(err)
                break

    # calculate confusion matrix
    cnf_matrix = confusion_matrix(labels_total, predictions_total)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    # calculate evaluate statics
    calculate_evaluate_statics(labels=labels_total, predictions=predictions_total)

    # plot precision recall curve
    plot_precision_recall_curve(labels=labels_total,
                                predictions_prob=predictions_prob_total,
                                class_nums=5)
    plt.show()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # test net
    assert ops.exists(args.dataset_dir)

    nsfw_eval_dataset(args.dataset_dir, args.weights_path)
