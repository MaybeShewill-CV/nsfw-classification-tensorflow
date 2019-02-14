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

import cv2
import glob
import glog as log
import tensorflow as tf
import numpy as np
import pprint

from config import global_config


CFG = global_config.cfg


def _write_example_tfrecords(example_paths, example_labels, tfrecords_path):
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
            _example_image = cv2.imread(_example_path, cv2.IMREAD_COLOR)
            _example_image = cv2.resize(_example_image,
                                        dsize=(CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT),
                                        interpolation=cv2.INTER_CUBIC)
            _example_image_raw = _example_image.tostring()

            _example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(CFG.TRAIN.IMG_HEIGHT),
                        'width': _int64_feature(CFG.TRAIN.IMG_WIDTH),
                        'depth': _int64_feature(3),
                        'label': _int64_feature(example_labels[_index]),
                        'image_raw': _bytes_feature(_example_image_raw)
                    }))
            _writer.write(_example.SerializeToString())

    log.info('Writing {:s} complete'.format(tfrecords_path))

    return


def _int64_feature(value):
    """

    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """

    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class NsfwDataProducer(object):
    """
    Convert raw image file into tfrecords
    """
    def __init__(self, dataset_dir):
        """

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
            process_pool.apply_async(func=_write_example_tfrecords,
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
            process_pool.apply_async(func=_write_example_tfrecords,
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
            process_pool.apply_async(func=_write_example_tfrecords,
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
        return ops.exists(self._drawing_image_dir) and ops.exists(self._hentai_image_dir) \
               and ops.exists(self._neural_image_dir) and ops.exists(self._porn_image_dir) \
               and ops.exists(self._sexy_image_dir)

    def _is_training_example_index_file_complete(self):
        """
        Check if the training example index file is complete
        :return:
        """
        return ops.exists(self._train_example_index_file_path) and \
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
    def __init__(self):
        """

        """
        pass


if __name__ == '__main__':
    """
    test code
    """

    # test nsfw data producer
    producer = NsfwDataProducer(dataset_dir='/media/baidu/Data/NSFW')

    producer.print_label_map()
    producer.generate_tfrecords(save_dir='/media/baidu/Data/NSFW/tfrecords', step_size=50)
