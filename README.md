# Nsfw-Classify-Tensorflow
NSFW classify model implemented with tensorflow. Use nsfw dataset provided here
https://github.com/alexkimxyz/nsfw_data_scraper Thanks for sharing the dataset
with us. You can find all the model details here. Don not hesitate to raise an
issue if you're confused with the model.

## Installation
This software has only been tested on ubuntu 16.04(x64). Here is the test environment
info

**OS**: Ubuntu 16.04 LTS

**GPU**: Two GTX 1070TI 

**CUDA**: cuda 9.0

**Tensorflow**: tensorflow 1.12.0

**OPENCV**: opencv 3.4.1

**NUMPY**: numpy 1.15.1

Other required package you may install them by

```
pip3 install -r requirements.txt
```

## Test model
In this repo I uploaded a pretrained dataset introduced as before, you need to 
download the pretrained weights file in folder REPO_ROOT_DIR/model/.

You can test a single image on the trained model as follows

```
cd REPO_ROOT_DIR
python tools/test_model.py --weights_path model/new_model/nsfw_cls.ckpt-100000
--image_path data/test_data/test_drawing.jpg
```

You can evaluate the model's performance on the nsfw dataset prepared in
advance as follows

```
cd REPO_ROOT_DIR
python tools/test_model.py --weights_path model/new_model/nsfw_cls.ckpt-100000
--dataset_dir PATH/TO/YOUR/NSFW_DATASET
```

The following part will show you how the dataset is well prepared

## Train your own model

#### Data Preparation
First you need to download all the origin nsfw data. Here is the 
[how_to_download_source_data](https://github.com/alexkimxyz/nsfw_data_scraper).
The training example should be organized like the what you can see in 
REPO_ROOT_DIR/data/nsfw_dataset_example. Then you should modified the 
REPO_ROOT_DIR/tools/make_nsfw_dataset.sh with your local nsfw dataset. Then excute
the dataset preparation script. That may take about one hour in my local machine.
You may enlarge the __C.TRAIN.CPU_MULTI_PROCESS_NUMS in config/global_config.py 
if you have a powerful cpu to accelerate the prepare process.

```
cd REPO_ROOT_DIR
bash tools/make_nsfw_dataset.sh
```

The image of each subclass will be split into three part according to the ratio
training : validation : test = 0.75 : 0.1 : 0.15. All the image will be convert
into tensorflow format record for efficient importing data pipline.

#### Train Model
The model support multi-gpu training. If you want to training the model on 
multiple gpus you need to first adjust the __C.TRAIN.GPU_NUM in config/global_config.py
file. Then excute the multi-gpu training procedure as follows:

```
cd REPO_ROOT_DIR
python tools/train_nsfw.py --dataset_dir PATH/TO/PREPARED_NSFW_DATASET --use_multi_gpu True
```

If you want to train the model from last snap shot you may excute following command:

```
cd REPO_ROOT_DIR
python tools/train_nsfw.py --dataset_dir PATH/TO/PREPARED_NSFW_DATASET 
--use_multi_gpu True --weights_path PATH/TO/YOUR/LAST_CKPT_FILE_PATH
```

You may set the --use_multi_gpu False then the whole training process will be excuted
on single gpu.

The main model's hyperparameter are as follows:

**iterations nums**: 120010

**learning rate**: 0.1

**batch size**: 32

**origin image size**: 256

**cropped image size**: 224

**training example nums**: 159477

**testing example nums**: 31895

**validation example nums**: 21266

The rest of the hyperparameter can be found [here](https://github.com/MaybeShewill-CV/nsfw-classification-tensorflow/blob/master/config/global_config.py).

You may monitor the training process using tensorboard tools

During my experiment the `train loss` drops as follows:  
![train_loss](/data/images/avg_train_loss.png)

The `train_top_1_error` rises as follows:  
![train_top_1_error](/data/images/avg_train_top1_error.png)

The `validation loss` drops as follows:  
![validation_loss](/data/images/avg_val_loss.png)

The `validation_top_1_error` rises as follows:  
![validation_top_1_error](/data/images/avg_val_top1_error.png)

#### Online demo
Since tensorflo-js is well supported the online deep learning is easy to deploy.
Here I have make a online demo to do local nsfw classification work. You may 
test here https://maybeshewill-cv.github.io/nsfw_classification The whole js work
can be found here https://github.com/MaybeShewill-CV/MaybeShewill-CV.github.io/tree/master/nsfw_classification
I have supplied a tool to convert the trained ckpt model file into tensorflow js
model file. Simply modify the file path and run the following script

```
cd ROOT_DIR
bash tools/convert_tfjs_model.sh
```

The online demo's example are as follows:
![online_demo](/data/images/online_demo.png)