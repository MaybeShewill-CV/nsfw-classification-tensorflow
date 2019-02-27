#!/usr/bin/env bash
tensorflowjs_converter --input_format=tf_saved_model --output_node_name=nsfw_cls_model/final_prediction \
--saved_model_tags=serve ./model/nsfw_export_saved_model ./model/nsfw_web_model