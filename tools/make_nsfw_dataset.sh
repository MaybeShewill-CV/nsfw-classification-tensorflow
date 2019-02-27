#!/usr/bin/env bash

# split the nsfw dataset and convert them into tfrecords
python nsfw_classification/data_provider/nsfw_data_feed_pipline.py \
--dataset_dir nsfw_classification/data/nsfw_dataset_example \
--tfrecords_dir nsfw_classification/data/nsfw_dataset_example/tfrecords
