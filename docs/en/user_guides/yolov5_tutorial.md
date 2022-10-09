# YOLOv5: From Start to Deployment

## Environment Setup

Note: Since this repository uses OpenMMLab 2.0, please create a new conda virtual environment to prevent conflicts with your existing repositories and projects of OpenMMLab 1.0.

```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch
# conda install pytorch torchvision cpuonly -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc1,<2.1.0"
mim install "mmdet>=3.0.0rc0,<3.1.0"
# for albumentations
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
# "-v" means verbose, or more output
# "-e" means install the project in editable mode, so any local modifications made to the code will take effect, eliminating the need to reinstall.
```

For more detailed information about environment configuration, please refer to [get_started](../get_started.md).

## Dataset Preparation

In this tutorial, the training dataset for MMYOLO is less than 40MB and is selected from the balloon dataset.

```shell
python tools/misc/download_dataset.py  --dataset-name balloon --save-dir data --unzip
python tools/dataset_converters/balloon2coco.py
```

After executing the above command, the balloon dataset will be downloaded in the `data` folder with the converted format we need. The `train.json` and `val.json` are the annotation files, both are in the coco format.

<div align=center>
<img src="https://cdn.vansin.top/img/20220912105312.png" alt="image"/>
</div>

## Config File Preparation

Create a new file called the `yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py` configuration file in the `configs/yolov5` folder, and copy the following content into it.

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

The above configuration is inherited from `./yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py`, and `data_root`, `metainfo`, `train_dataloader`, `val_dataloader`, `num_classes` and other configurations are updated according to the balloon data we are using.
We set the `interval` of the logger to 1 because each iteration of the `interval` will output a loss-related log, and the balloon dataset we use is relatively tiny. We will not see the loss-related output if the `interval` is too large.

## Training

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py
```

After executing the above training command, the `work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon` folder will be automatically generated. Both the weight and the training configuration files will be saved in this folder.

<div align=center>
<img src="https://cdn.vansin.top/img/20220913213846.png" alt="image"/>
</div>