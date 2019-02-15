#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-2-15 下午6:56
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : nsfw_classification_net.py
# @IDE: PyCharm
"""
NSFW Classification Net Model
"""
import tensorflow as tf

from nsfw_model import cnn_basenet
from config import  global_config


CFG = global_config.cfg


class NSFWNet(cnn_basenet.CNNBaseModel):
    """
    nsfw classification net
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(NSFWNet, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def _bn_relu_conv_layer(self, input_tensor, k_size, out_dims, stride, name):
        """

        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param stride:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):

            bn = self.layerbn(inputdata=input_tensor, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

            conv = self.conv2d(inputdata=relu,
                               out_channel=out_dims,
                               kernel_size=k_size,
                               stride=stride,
                               weight_decay=CFG.TRAIN.WEIGHT_DECAY,
                               use_bias=False,
                               name='conv')
        return conv

    def _conv_bn_relu_layer(self, input_tensor, k_size, out_dims, stride, name):
        """

        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param stride:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            conv = self.conv2d(inputdata=input_tensor,
                               out_channel=out_dims,
                               kernel_size=k_size,
                               stride=stride,
                               weight_decay=CFG.TRAIN.WEIGHT_DECAY,
                               use_bias=False,
                               name='conv')
            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _residual_block(self, input_tensor, output_channel, first_block=False):
        """
        Defines a residual block in ResNet
        :param input_tensor: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.
        """
        input_channel = input_tensor.get_shape().as_list()[-1]

        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                conv_1 = self.conv2d(inputdata=input_tensor,
                                     out_channel=output_channel,
                                     kernel_size=3,
                                     stride=1,
                                     weight_decay=CFG.TRAIN.WEIGHT_DECAY,
                                     use_bias=False,
                                     name='conv_1')
            else:
                conv_1 = self._bn_relu_conv_layer(input_tensor=input_tensor,
                                                  k_size=3,
                                                  out_dims=output_channel,
                                                  stride=stride,
                                                  name='conv_1')

        with tf.variable_scope('conv2_in_block'):
            conv_2 = self._bn_relu_conv_layer(input_tensor=conv_1,
                                              k_size=3,
                                              out_dims=output_channel,
                                              stride=1,
                                              name='conv_2')

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        # depth of input layers
        if increase_dim is True:
            pooled_input = self.avgpooling(inputdata=input_tensor,
                                           kernel_size=2,
                                           stride=2,
                                           padding='VALID',
                                           name='avg_pool')
            padded_input = tf.pad(pooled_input,
                                  [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
        else:
            padded_input = input_tensor

        output = conv_2 + padded_input

        return output

    def inference(self, input_tensor, residual_blocks_nums, name, reuse=False):
        """
        The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n + 1 = 6n + 2
        :param input_tensor: 4D tensor
        :param residual_blocks_nums: num_residual_blocks
        :param name: net name
        :param reuse: To build train graph, reuse=False. To build validation graph and share weights
        with train graph, resue=True
        :return: last layer in the network. Not softmax-ed
        """
        layers = []

        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            with tf.variable_scope('conv0', reuse=reuse):
                conv_0 = self._conv_bn_relu_layer(input_tensor=input_tensor,
                                                  k_size=3,
                                                  out_dims=16,
                                                  stride=1,
                                                  name='conv_0')
                layers.append(conv_0)

            for i in range(residual_blocks_nums):
                with tf.variable_scope('conv_1_{:d}'.format(i), reuse=reuse):
                    if i == 0:
                        conv_1 = self._residual_block(layers[-1], 16, first_block=True)
                    else:
                        conv_1 = self._residual_block(layers[-1], 16, first_block=False)

                    layers.append(conv_1)

            for i in range(residual_blocks_nums):
                with tf.variable_scope('conv_2_{:d}'.format(i), reuse=reuse):
                    conv_2 = self._residual_block(layers[-1], 32)

                    layers.append(conv_2)

            for i in range(residual_blocks_nums):
                with tf.variable_scope('conv_3_{:d}'.format(i), reuse=reuse):
                    conv_3 = self._residual_block(layers[-1], 64)

                    layers.append(conv_3)

            with tf.variable_scope('fc', reuse=reuse):

                bn = self.layerbn(inputdata=layers[-1], is_training=self._is_training, name='bn')

                relu = self.relu(inputdata=bn, name='relu')

                global_pool = self.globalavgpooling(inputdata=relu, name='global_avg_pool')

                final_logits = self.fullyconnect(inputdata=global_pool,
                                                 out_dim=CFG.TRAIN.CLASSES_NUMS,
                                                 weight_decay=CFG.TRAIN.WEIGHT_DECAY,
                                                 w_init=tf.initializers.variance_scaling(distribution='uniform'),
                                                 b_init=tf.zeros_initializer(),
                                                 use_bias=True,
                                                 name='final_logits')

                layers.append(final_logits)

        return final_logits

    def compute_loss(self, input_tensor, labels, residual_blocks_nums, name, reuse=False):
        """

        :param input_tensor:
        :param labels:
        :param residual_blocks_nums:
        :param name:
        :param reuse:
        :return:
        """
        labels = tf.cast(labels, tf.int64)

        inference_logits = self.inference(input_tensor=input_tensor,
                                          residual_blocks_nums=residual_blocks_nums,
                                          name=name,
                                          reuse=reuse)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=inference_logits,
                                                                       labels=labels,
                                                                       name='cross_entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        total_loss = tf.add_n([loss] + regu_losses)

        return total_loss


if __name__ == '__main__':
    """
    test code
    """
    image_tensor = tf.placeholder(shape=[16, 224, 224, 3], dtype=tf.float32)
    label_tensor = tf.placeholder(shape=[16], dtype=tf.int32)

    net = NSFWNet(phase=tf.constant('train', dtype=tf.string))

    loss = net.compute_loss(input_tensor=image_tensor,
                            labels=label_tensor,
                            residual_blocks_nums=CFG.NET.RES_BLOCKS_NUMS,
                            name='net',
                            reuse=False)
    loss_val = net.compute_loss(input_tensor=image_tensor,
                                labels=label_tensor,
                                residual_blocks_nums=CFG.NET.RES_BLOCKS_NUMS,
                                name='net',
                                reuse=True)

    inference_logits = net.inference(input_tensor=image_tensor,
                                     residual_blocks_nums=CFG.NET.RES_BLOCKS_NUMS,
                                     name='net',
                                     reuse=True)

    print(loss.get_shape().as_list())
    print(loss_val.get_shape().as_list())
    print(inference_logits.get_shape().as_list())
