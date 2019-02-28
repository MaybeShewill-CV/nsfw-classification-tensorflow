#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-2-15 下午6:56
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/MaybeShewill-CV.github.io
# @File    : nsfw_classification_net.py
# @IDE: PyCharm
"""
NSFW Classification Net Model
"""
import tensorflow as tf

from nsfw_model import cnn_basenet
from config import global_config


CFG = global_config.cfg


class NSFWNet(cnn_basenet.CNNBaseModel):
    """
    nsfw classification net
    """
    def __init__(self, phase, resnet_size=CFG.NET.RESNET_SIZE):
        """

        :param phase:
        """
        super(NSFWNet, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()
        self._need_summary_feats_map = CFG.NET.NEED_SUMMARY_FEATS_MAP
        self._resnet_size = resnet_size
        self._block_sizes = self._get_block_sizes(self._resnet_size)
        self._block_strides = [1, 2, 2, 2]

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    @staticmethod
    def _get_block_sizes(resnet_size):
        """
        Retrieve the size of each block_layer in the ResNet model.
        The number of block layers used for the Resnet model varies according
        to the size of the model. This helper grabs the layer set we want, throwing
        an error if a non-standard size has been selected.
        Args:
          resnet_size: The number of convolutional layers needed in the model.
        Returns:
          A list of block sizes to use in building the model.
        Raises:
          KeyError: if invalid resnet_size is received.
        """
        choices = {
            32: [3, 4, 6, 3],
            50: [3, 4, 6, 3]
        }

        try:
            return choices[resnet_size]
        except KeyError:
            err = ('Could not find layers for selected Resnet size.\n'
                   'Size received: {}; sizes allowed: {}.'.format(resnet_size, choices.keys()))
            raise ValueError(err)

    @staticmethod
    def _feature_map_summary(input_tensor, slice_nums, axis):
        """
        summary feature map
        :param input_tensor: A Tensor
        :return: Add histogram summary and scalar summary of the sparsity of the tensor
        """
        tensor_name = input_tensor.op.name

        split = tf.split(input_tensor, num_or_size_splits=slice_nums, axis=axis)
        for i in range(slice_nums):
            tf.summary.image(tensor_name + "/feature_maps_" + str(i), split[i])

    def _fixed_padding(self, inputs, kernel_size, name):
        """Pads the input along the spatial dimensions independently of input size.
        Args:
          inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                       Should be a positive integer.
          name:
        Returns:
          A tensor with the same format as the input with the data either intact
          (if kernel_size == 1) or padded (if kernel_size > 1).
        """
        with tf.variable_scope(name_or_scope=name):
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg

            padded_inputs = self.pad(inputdata=inputs,
                                     paddings=[[0, 0], [pad_beg, pad_end],
                                              [pad_beg, pad_end], [0, 0]],
                                     name='pad')
        return padded_inputs

    def _conv2d_fixed_padding(self, inputs, kernel_size, output_dims, strides, name):
        """

        :param inputs:
        :param kernel_size:
        :param output_dims:
        :param strides:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            if strides > 1:
                inputs = self._fixed_padding(inputs, kernel_size, name='fix_padding')

            result = self.conv2d(inputdata=inputs, out_channel=output_dims, kernel_size=kernel_size,
                                 stride=strides, padding=('SAME' if strides == 1 else 'VALID'),
                                 use_bias=False, name='conv')

        return result

    def _process_image_input_tensor(self, input_image_tensor, kernel_size,
                                    conv_stride, output_dims, pool_size, pool_stride):
        """
        Resnet entry
        :param input_image_tensor:
        :param kernel_size:
        :param conv_stride:
        :param output_dims:
        :param pool_size:
        :param pool_stride:
        :return:
        """
        inputs = self._conv2d_fixed_padding(
            inputs=input_image_tensor, kernel_size=kernel_size,
            strides=conv_stride, output_dims=output_dims, name='initial_conv_pad')
        inputs = tf.identity(inputs, 'initial_conv')

        inputs = self.maxpooling(inputdata=inputs, kernel_size=pool_size,
                                 stride=pool_stride, padding='SAME',
                                 name='initial_max_pool')

        return inputs

    def _resnet_block_fn(self, input_tensor, kernel_size, stride,
                         output_dims, name, projection_shortcut=None):
        """
        A single block for ResNet v2, without a bottleneck.
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
        :param input_tensor:
        :param kernel_size:
        :param stride:
        :param output_dims:
        :param name:
        :param projection_shortcut:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            shortcut = input_tensor
            inputs = self.layerbn(inputdata=input_tensor, is_training=self._is_training, name='bn_1')
            inputs = self.relu(inputdata=inputs, name='relu_1')

            if projection_shortcut is not None:
                shortcut = projection_shortcut(inputs)

            inputs = self._conv2d_fixed_padding(
                inputs=inputs, output_dims=output_dims,
                kernel_size=kernel_size, strides=stride, name='conv_pad_1')

            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_2')
            inputs = self.relu(inputdata=inputs, name='relu_2')
            inputs = self._conv2d_fixed_padding(
                inputs=inputs, output_dims=output_dims, kernel_size=kernel_size, strides=1, name='conv_pad_2')

        return inputs + shortcut

    def _resnet_block_layer(self, input_tensor, kernel_size, stride, block_nums, output_dims, name):
        """

        :param input_tensor:
        :param kernel_size:
        :param stride:
        :param block_nums:
        :param name:
        :return:
        """
        def projection_shortcut(_inputs):
            return self._conv2d_fixed_padding(
                inputs=_inputs, output_dims=output_dims, kernel_size=1,
                strides=stride, name='projection_shortcut')

        with tf.variable_scope(name):
            inputs = self._resnet_block_fn(input_tensor=input_tensor,
                                           kernel_size=kernel_size,
                                           output_dims=output_dims,
                                           projection_shortcut=projection_shortcut,
                                           stride=stride,
                                           name='init_block_fn')

            for index in range(1, block_nums):
                inputs = self._resnet_block_fn(input_tensor=inputs,
                                               kernel_size=kernel_size,
                                               output_dims=output_dims,
                                               projection_shortcut=None,
                                               stride=1,
                                               name='block_fn_{:d}'.format(index))
        return inputs

    def inference(self, input_tensor, name, reuse=False):
        """
        The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n + 1 = 6n + 2
        :param input_tensor: 4D tensor
        :param name: net name
        :param reuse: To build train graph, reuse=False. To build validation graph and share weights
        with train graph, resue=True
        :return: last layer in the network. Not softmax-ed
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):

            if self._need_summary_feats_map:
                self._feature_map_summary(input_tensor=input_tensor, slice_nums=1, axis=-1)

            # first layer process
            inputs = self._process_image_input_tensor(input_image_tensor=input_tensor,
                                                      kernel_size=7,
                                                      conv_stride=2,
                                                      output_dims=64,
                                                      pool_size=3,
                                                      pool_stride=2)
            if self._need_summary_feats_map:
                self._feature_map_summary(input_tensor=inputs, slice_nums=64, axis=-1)

            for index, block_nums in enumerate(self._block_sizes):
                output_dims = 64 * (2 ** index)

                inputs = self._resnet_block_layer(input_tensor=inputs,
                                                  kernel_size=3,
                                                  output_dims=output_dims,
                                                  block_nums=block_nums,
                                                  stride=self._block_strides[index],
                                                  name='resnet_block_layer_{:d}'.format(index + 1))

                if self._need_summary_feats_map:
                    self._feature_map_summary(input_tensor=inputs, slice_nums=output_dims, axis=-1)

            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_after_block_layer')
            inputs = self.relu(inputdata=inputs, name='relu_after_block_layer')

            inputs = tf.reduce_mean(input_tensor=inputs, axis=[1, 2], keepdims=True, name='final_reduce_mean')
            inputs = tf.squeeze(input=inputs, axis=[1, 2], name='final_squeeze')

            final_logits = self.fullyconnect(inputdata=inputs, out_dim=CFG.TRAIN.CLASSES_NUMS,
                                             use_bias=False, name='final_logits')

        return final_logits

    def compute_loss(self, input_tensor, labels, name, reuse=False):
        """

        :param input_tensor:
        :param labels:
        :param name:
        :param reuse:
        :return:
        """
        labels = tf.cast(labels, tf.int64)

        inference_logits = self.inference(input_tensor=input_tensor,
                                          name=name,
                                          reuse=reuse)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=inference_logits,
                                                                       labels=labels,
                                                                       name='cross_entropy_per_example')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

        l2_loss = CFG.TRAIN.WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(tf.cast(vv, tf.float32)) for vv in tf.trainable_variables()
             if 'bn' not in vv.name])

        total_loss = cross_entropy_loss + l2_loss

        return total_loss


if __name__ == '__main__':
    """
    test code
    """
    image_tensor = tf.placeholder(shape=[16, 256, 256, 3], dtype=tf.float32)
    label_tensor = tf.placeholder(shape=[16], dtype=tf.int32)

    net = NSFWNet(phase=tf.constant('train', dtype=tf.string))

    loss = net.compute_loss(input_tensor=image_tensor,
                            labels=label_tensor,
                            name='net',
                            reuse=False)
    loss_val = net.compute_loss(input_tensor=image_tensor,
                                labels=label_tensor,
                                name='net',
                                reuse=True)

    logits = net.inference(input_tensor=image_tensor,
                           name='net',
                           reuse=True)

    print(loss.get_shape().as_list())
    print(loss_val.get_shape().as_list())
    print(logits.get_shape().as_list())
