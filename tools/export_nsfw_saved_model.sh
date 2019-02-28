#!/usr/bin/env bash

python tools/export_saved_model.py --ckpt_path model/nsfw_cls/nsfw_cls_2019-02-27-18-46-28.ckpt-160000 \
--export_dir ./model/nsfw_export_saved_model