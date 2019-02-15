#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-2-14 下午5:43
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : nsfw_data_feed_pipline.py
# @IDE: PyCharm
"""
nsfw数据feed pipline
"""
import os
import os.path as ops
import random
import multiprocessing

import glob
import glog as log
import tensorflow as tf
import pprint

from config import global_config
from data_provider import tf_io_pipline_tools


CFG = global_config.cfg


class NsfwDataProducer(object):
    """
    Convert raw image file into tfrecords
    """
    def __init__(self, dataset_dir):
        """

        :param dataset_dir:
        """
        self._label_map = {
            'drawing': 0,
            'hentai': 1,
            'neural': 2,
            'porn': 3,
            'sexy': 4
        }

        self._dataset_dir = dataset_dir

        self._drawing_image_dir = ops.join(dataset_dir, 'drawing')
        self._hentai_image_dir = ops.join(dataset_dir, 'hentai')
        self._neural_image_dir = ops.join(dataset_dir, 'neural')
        self._porn_image_dir = ops.join(dataset_dir, 'porn')
        self._sexy_image_dir = ops.join(dataset_dir, 'sexy')

        self._train_example_index_file_path = ops.join(self._dataset_dir, 'train.txt')
        self._test_example_index_file_path = ops.join(self._dataset_dir, 'test.txt')
        self._val_example_index_file_path = ops.join(self._dataset_dir, 'val.txt')

        if not self._is_source_data_complete():
            raise ValueError('Source image data is not complete, '
                             'please check if one of the image folder is not exist')

        if not self._is_training_example_index_file_complete():
            self._generate_training_example_index_file()

    def print_label_map(self):
        """
        Print label map which show the labels information
        :return:
        """
        pprint.pprint(self._label_map)

    def generate_tfrecords(self, save_dir, step_size=10000):
        """
        Generate tensorflow records file
        :param save_dir:
        :param step_size: generate a tfrecord every step_size examples
        :return:
        """
        def _read_training_example_index_file(_index_file_path):

            assert ops.exists(_index_file_path)

            _example_path_info = []
            _example_label_info = []

            with open(_index_file_path, 'r') as _file:
                for _line in _file:
                    _example_info = _line.rstrip('\r').rstrip('\n').split(' ')
                    _example_path_info.append(_example_info[0])
                    _example_label_info.append(int(_example_info[1]))

            return _example_path_info, _example_label_info

        def _split_writing_tfrecords_task(_example_paths, _example_labels, _flags='train'):

            _split_example_paths = []
            _split_example_labels = []
            _split_tfrecords_save_paths = []

            for i in range(0, len(_example_paths), step_size):
                _split_example_paths.append(_example_paths[i:i + step_size])
                _split_example_labels.append(_example_labels[i:i + step_size])

                if i + step_size > len(_example_paths):
                    _split_tfrecords_save_paths.append(
                        ops.join(save_dir, '{:s}_{:d}_{:d}.tfrecords'.format(_flags, i, len(_example_paths))))
                else:
                    _split_tfrecords_save_paths.append(
                        ops.join(save_dir, '{:s}_{:d}_{:d}.tfrecords'.format(_flags, i, i + step_size)))

            return _split_example_paths, _split_example_labels, _split_tfrecords_save_paths

        # make save dirs
        os.makedirs(save_dir, exist_ok=True)

        # set process pool
        process_pool = multiprocessing.Pool(processes=4)

        # generate training example tfrecords
        log.info('Generating training example tfrecords')

        train_example_paths, train_example_labels = _read_training_example_index_file(
            self._train_example_index_file_path)
        train_example_paths_split, train_example_labels_split, train_tfrecords_save_paths = \
            _split_writing_tfrecords_task(train_example_paths, train_example_labels, _flags='train')

        for index, example_paths in enumerate(train_example_paths_split):
            process_pool.apply_async(func=tf_io_pipline_tools.write_example_tfrecords,
                                     args=(example_paths,
                                           train_example_labels_split[index],
                                           train_tfrecords_save_paths[index],))

        process_pool.close()
        process_pool.join()

        log.info('Generate training example tfrecords complete')

        # set process pool
        process_pool = multiprocessing.Pool(processes=4)

        # generate val example tfrecords
        log.info('Generating validation example tfrecords')

        val_example_paths, val_example_labels = _read_training_example_index_file(
            self._val_example_index_file_path)
        val_example_paths_split, val_example_labels_split, val_tfrecords_save_paths = \
            _split_writing_tfrecords_task(val_example_paths, val_example_labels, _flags='val')

        for index, example_paths in enumerate(val_example_paths_split):
            process_pool.apply_async(func=tf_io_pipline_tools.write_example_tfrecords,
                                     args=(example_paths,
                                           val_example_labels_split[index],
                                           val_tfrecords_save_paths[index],))

        process_pool.close()
        process_pool.join()

        log.info('Generate validation example tfrecords complete')

        # set process pool
        process_pool = multiprocessing.Pool(processes=4)

        # generate test example tfrecords
        log.info('Generating testing example tfrecords')

        test_example_paths, test_example_labels = _read_training_example_index_file(
            self._test_example_index_file_path)
        test_example_paths_split, test_example_labels_split, test_tfrecords_save_paths = \
            _split_writing_tfrecords_task(test_example_paths, test_example_labels, _flags='test')

        for index, example_paths in enumerate(test_example_paths_split):
            process_pool.apply_async(func=tf_io_pipline_tools.write_example_tfrecords,
                                     args=(example_paths,
                                           test_example_labels_split[index],
                                           test_tfrecords_save_paths[index],))

        process_pool.close()
        process_pool.join()

        log.info('Generate testing example tfrecords complete')

        return

    def _is_source_data_complete(self):
        """
        Check if source data complete
        :return:
        """
        return \
            ops.exists(self._drawing_image_dir) and ops.exists(self._hentai_image_dir) \
            and ops.exists(self._neural_image_dir) and ops.exists(self._porn_image_dir) \
            and ops.exists(self._sexy_image_dir)

    def _is_training_example_index_file_complete(self):
        """
        Check if the training example index file is complete
        :return:
        """
        return \
            ops.exists(self._train_example_index_file_path) and \
            ops.exists(self._test_example_index_file_path) and \
            ops.exists(self._val_example_index_file_path)

    def _generate_training_example_index_file(self):
        """
        Generate training example index file, split source file into 0.75, 0.15, 0.1 for training,
        testing and validation. Each image folder are processed separately
        :return:
        """
        def _process_single_training_folder(_folder_dir):

            _folder_label_name = ops.split(_folder_dir)[1]
            _folder_label_index = self._label_map[_folder_label_name]

            _source_image_paths = glob.glob('{:s}/*'.format(_folder_dir))

            return ['{:s} {:d}\n'.format(s, _folder_label_index) for s in _source_image_paths]

        def _split_training_examples(_example_info):

            random.shuffle(_example_info)

            _example_nums = len(_example_info)

            _train_example_info = _example_info[:int(_example_nums * 0.75)]
            _test_example_info = _example_info[int(_example_nums * 0.75):int(_example_nums * 0.9)]
            _val_example_info = _example_info[int(_example_nums * 0.9):]

            return _train_example_info, _test_example_info, _val_example_info

        train_example_info = []
        test_example_info = []
        val_example_info = []

        for example_dir in [self._drawing_image_dir, self._hentai_image_dir,
                            self._neural_image_dir, self._porn_image_dir,
                            self._sexy_image_dir]:
            _train_tmp_info, _test_tmp_info, _val_tmp_info = \
                _split_training_examples(_process_single_training_folder(example_dir))

            train_example_info.extend(_train_tmp_info)
            test_example_info.extend(_test_tmp_info)
            val_example_info.extend(_val_tmp_info)

        random.shuffle(train_example_info)
        random.shuffle(test_example_info)
        random.shuffle(val_example_info)

        with open(ops.join(self._dataset_dir, 'train.txt'), 'w') as file:
            file.write(''.join(train_example_info))

        with open(ops.join(self._dataset_dir, 'test.txt'), 'w') as file:
            file.write(''.join(test_example_info))

        with open(ops.join(self._dataset_dir, 'val.txt'), 'w') as file:
            file.write(''.join(val_example_info))

        log.info('Generate training example index file complete')

        return


class NsfwDataFeeder(object):
    """
    Read training examples from tfrecords for nsfw model
    """
    def __init__(self, dataset_dir, flags='train'):
        """

        :param dataset_dir:
        :param flags:
        """
        self._dataset_dir = dataset_dir

        self._tfrecords_dir = ops.join(dataset_dir, 'tfrecords')
        if not ops.exists(self._tfrecords_dir):
            raise ValueError('{:s} not exist, please check again'.format(self._tfrecords_dir))

        self._dataset_flags = flags.lower()
        if self._dataset_flags not in ['train', 'test', 'val']:
            raise ValueError('flags of the data feeder should be \'train\', \'test\', \'val\'')

    def inputs(self, batch_size, num_epochs):
        """
        dataset feed pipline input
        :param batch_size:
        :param num_epochs:
        :return: A tuple (images, labels), where:
                    * images is a float tensor with shape [batch_size, H, W, C]
                      in the range [-0.5, 0.5].
                    * labels is an int32 tensor with shape [batch_size] with the true label,
                      a number in the range [0, CLASS_NUMS).
        """
        if not num_epochs:
            num_epochs = None

        tfrecords_file_paths = glob.glob('{:s}/{:s}*.tfrecords'.format(self._tfrecords_dir, self._dataset_flags))
        random.shuffle(tfrecords_file_paths)

        with tf.name_scope('input_tensor'):

            # TFRecordDataset opens a binary file and reads one record at a time.
            # `tfrecords_file_paths` could also be a list of filenames, which will be read in order.
            dataset = tf.data.TFRecordDataset(tfrecords_file_paths)

            # The map transformation takes a function and applies it to every element
            # of the dataset.
            dataset = dataset.map(tf_io_pipline_tools.decode)
            dataset = dataset.map(tf_io_pipline_tools.augment)
            dataset = dataset.map(tf_io_pipline_tools.normalize)

            # The shuffle transformation uses a finite-sized buffer to shuffle elements
            # in memory. The parameter is the number of elements in the buffer. For
            # completely uniform shuffling, set the parameter to be the same as the
            # number of elements in the dataset.
            dataset = dataset.shuffle(buffer_size=5000)

            # repeat num epochs
            dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(batch_size)

            iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()


if __name__ == '__main__':
    """
    test code
    """

    # test nsfw data producer
    producer = NsfwDataProducer(dataset_dir='/media/baidu/Data/NSFW')

    producer.print_label_map()
    producer.generate_tfrecords(save_dir='/media/baidu/Data/NSFW/tfrecords', step_size=10000)

    # test nsfw data feeder
    feeder = NsfwDataFeeder(dataset_dir='/media/baidu/Data/NSFW', flags='train')
    images, labels = feeder.inputs(16, 1)

    image_shape = tf.stack([16, 224, 224, 3])
    images = tf.reshape(shape=image_shape, tensor=images)
