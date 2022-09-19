# From Getting Started to Deployment tutorial with YOLOv5

## Installation

NOTICE: MMYOLO is based on OpenMMLab 2.0, it is highly recommended to create a new conda virtual environment to prevent conflicts the repository already installed by OpenMMLab 1.0. 

```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch
# conda install pytorch torchvision cpuonly -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc1"
mim install "mmdet>=3.0.0rc0"

git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
pip install -v -e .
```

Please see [get_started](../get_started.md) for detailed installation instructions.

## Dataset Preparation

We provide the ballon dataset, a small dataset of less than 40MB in size, as the learning dataset of MMYOLO.

```shell
python tools/misc/download_dataset.py --dataset-name balloon --save-dir data --unzip
python tools/dataset_converters/balloon2coco.py
```

After executing the above command, the command automatically downloads the dataset, converts the format of annotations. The balloon dataset is ready in the `data` folder. The `train.json` and `val.json` are the annotation files in coco format. 

![](https://cdn.vansin.top/img/20220912105312.png)

## Config file Preparation

Create a new `yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py` configuration file in the folder of `configs/yolov5`, and copy the following content into the configuration file. 

```python
_base_ = './yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = 'data/balloon/'

train_batch_size_per_gpu = 4
train_num_workers = 2

metainfo = {
    'CLASSES': ('balloon', ),
    'PALETTE': [
        (220, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='val/'),
        ann_file='val.json'))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val.json')

test_evaluator = val_evaluator

model = dict(bbox_head=dict(head_module=dict(num_classes=1)))

default_hooks = dict(logger=dict(interval=1))

```

It is worth noticing that the above configuration file is inherited from `./yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py`, and `data_root`, `metainfo`, `train_dataloader`, `val_dataloader`, `num_classes` and other configurations are updated according to the characteristics of balloon data.

The reason why we set the interval of the logger to 1 is that each interval iteration will output a loss-related log, and the balloon data set we choose is relatively small, and if the interval is too large, we will not see the output of the loss-related log.

## Training

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py
```

Run the above training command, the `work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon` folder will be automatically generated, and the weight file and the training configuration file will be saved in this folder.

![](https://cdn.vansin.top/img/20220913213846.png)

### Resume training after interruptions

If the training stops midway, add `--resume` at the end of the training command, and the program will automatically load the latest weight file from `work_dirs` to resume training.

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py --resume
```

### Finetuning from pretrained models

NOTICE: It is highly recommended that finetuning from large datasets, such as COCO, can significantly boost the performance of overall network. 
In this example, compared with training from scratch, the pretrained model with finetuning outperforms with a significant margin. (Over 30+ mAP boost than training from scratch).

1. Download the COCO dataset pre-trained weights

```shell
cd mmyolo
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth
```

2. Load the pretrained model for training

```shell
cd mmyolo
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py --cfg-options load_from='yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'
```

3. Visualize training parameters

This tutorial uses wandb to show the visualization of loss and other data. Register on the official website of wandb and get the API Keys of wandb at https://wandb.ai/settings

![](https://cdn.vansin.top/img/20220913212628.png)

```shell
pip install wandb
# After running wandb login, enter the API Keys obtained above, and the login is successful.
wandb login
```
Add wandb configuration in `configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py`

```python
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
```

Re-run the training command to see data visualizations such as loss, learning rate, and coco/bbox_mAP in the web link prompted on the command line.

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py
```

![](https://cdn.vansin.top/img/20220913213221.png)

### Model inference

```shell
python tools/test.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon/epoch_300.pth --show-dir show_results
```

Run the above inference command, the inference result picture will be automatically saved to the `work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon/{timestamp}/show_results` folder. The following is one of the result pictures, the left picture is the actual annotation, and the right picture is the model inference result.

![result_img](https://user-images.githubusercontent.com/27466624/190913272-f99709e5-c798-46b8-aede-30f4e91683a3.jpg)
