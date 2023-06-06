# 15 minutes to get started with MMYOLO instance segmentation

Instance segmentation is a task in computer vision that aims to segment each object in an image and assign each object a unique identifier.

Unlike semantic segmentation, instance segmentation not only segments out different categories in an image, but also separates different instances of the same category.

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/87774050/6fd6316f-d78d-48a5-a413-86e7a74583fd" alt="Instance Segmentation" width="100%"/>
</div>

Taking the downloadable balloon dataset as an example, I will guide you through a 15-minute easy introduction to MMYOLO instance segmentation. The entire process includes the following steps:

- [Installation](#installation)
- [Dataset](#dataset)
- [Config](#config)
- [Training](#training)
- [Testing](#testing)
- [EasyDeploy](#easydeploy-deployment)

In this tutorial, we will use YOLOv5-s as an example. For the demo configuration of the balloon dataset with other YOLO series algorithms, please refer to the corresponding algorithm configuration folder.

## Installation

Assuming you've already installed Conda in advance, then install PyTorch using the following commands.

```{note}
Note: Since this repo uses OpenMMLab 2.0, it is better to create a new conda virtual environment to prevent conflicts with the repo installed in OpenMMLab 1.0.
```

```shell
conda create -n mmyolo python=3.8 -y
conda activate mmyolo
# If you have GPU
conda install pytorch torchvision -c pytorch
# If you only have CPU
# conda install pytorch torchvision cpuonly -c pytorch
```

Install MMYOLO and dependency libraries using the following commands.

```shell
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
pip install -U openmim
mim install -r requirements/mminstall.txt
# Install albumentations
mim install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

For details about how to configure the environment, see [Installation and verification](./installation.md).

## Dataset

The Balloon dataset is a single-class dataset that consists of 74 images and includes annotated information required for training. Here is an example image from the dataset:

<div align=center>
<img src="https://user-images.githubusercontent.com/87774050/236993643-f581b087-9231-48a5-810b-97a5f31abe63.png" alt="balloon dataset"/>
</div>

You can download and use it directly by the following command:

```shell
python tools/misc/download_dataset.py --dataset-name balloon --save-dir ./data/balloon --unzip --delete
python ./tools/dataset_converters/balloon2coco.py
```

The data for the MMYOLO project is located in the MMYOLO project directory. The `train.json` and `val.json` files store the annotations in COCO format, while the `data/balloon/train` and `data/balloon/val` directories contain all the images for the dataset.

## Config

Taking YOLOv5 algorithm as an example, considering the limited GPU memory of users, we need to modify some default training parameters to make them run smoothly. The key parameters to be modified are as follows:

- YOLOv5 is an Anchor-Based algorithm, and different datasets need to calculate suitable anchors adaptively.
- The default config uses 8 GPUs with a batch size of 16 per GPU. Now change it to a single GPU with a batch size of 12.
- In principle, the learning rate should be linearly scaled accordingly when the batch size is changed, but actual measurements have found that this is not necessary.

To perform the specific operation, create a new configuration file named `yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py` in the `configs/yolov5/ins_seg` folder. For convenience, we have already provided this configuration file. Copy the following contents into the configuration file.

```python
_base_ = './yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance.py'  # noqa

data_root = 'data/balloon/' # dataset root
# Training set annotation file of json path
train_ann_file = 'train.json'
train_data_prefix = 'train/'  # Dataset prefix
# Validation set annotation file of json path
val_ann_file = 'val.json'
val_data_prefix = 'val/'
metainfo = {
    'classes': ('balloon', ), # dataset category name
    'palette': [
        (220, 20, 60),
    ]
}
num_classes = 1
# Set batch size to 4
train_batch_size_per_gpu = 4
# dataloader num workers
train_num_workers = 2
log_interval = 1
#####################
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=train_data_prefix),
        ann_file=train_ann_file))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator
default_hooks = dict(logger=dict(interval=log_interval))
#####################

model = dict(bbox_head=dict(head_module=dict(num_classes=num_classes)))
```

The above configuration inherits from `yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance.py` and updates configurations such as `data_root`, `metainfo`, `train_dataloader`, `val_dataloader`, `num_classes`, etc., based on the characteristics of the balloon dataset.

## Training

```shell
python tools/train.py configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py
```

After running the training command mentioned above, the folder `work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance` will be automatically generated. The weight files and the training configuration file for this session will be saved in this folder. On a lower-end GPU like the GTX 1660, the entire training process will take approximately 30 minutes.

<div align=center>
<img src="https://user-images.githubusercontent.com/87774050/236995027-00a16a9e-2a2d-44cc-be8a-e2c8a36ff77f.png" alt="image"/>
</div>

The performance on `val.json` is as follows:

```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.330
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.509
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.317
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.103
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.417
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.317
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.525
```

The above performance is obtained by printing using the COCO API, where -1 indicates the absence of objects of that scale.

### Some Notes

The key warnings are printed during training:

- You are using `YOLOv5Head` with num_classes == 1. The loss_cls will be 0. This is a normal phenomenon.

The warning is because the `num_classes` currently trained is 1, the loss of the classification branch is always 0 according to the community of the YOLOv5 algorithm, which is a normal phenomenon.

### Training is resumed after the interruption

If you stop training, you can add `--resume` to the end of the training command and the program will automatically resume training with the latest weights file from `work_dirs`.

```shell
python tools/train.py configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py --resume
```

### Save GPU memory strategy

The above config requires about 3G RAM, so if you don't have enough, consider turning on mixed-precision training

```shell
python tools/train.py configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py --amp
```

### Training visualization

MMYOLO currently supports local, TensorBoard, WandB and other back-end visualization. The default is to use local visualization, and you can switch to WandB and other real-time visualization of various indicators in the training process.

#### 1 WandB

WandB visualization need registered in website, and in the https://wandb.ai/settings for wandb API Keys.

<div align=center>
<img src="https://cdn.vansin.top/img/20220913212628.png" alt="image"/>
</div>

```shell
pip install wandb
# After running wandb login, enter the API Keys obtained above, and the login is successful.
wandb login
```

Add the wandb config at the end of config file we just created: `configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py`.

```python
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
```

Running the training command and you will see the loss, learning rate, and coco/bbox_mAP visualizations in the link.

```shell
python tools/train.py configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py
```

#### 2 Tensorboard

Install Tensorboard package using the following command:

```shell
pip install tensorboard
```

Add the `tensorboard` config at the end of config file we just created: `configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py`.

```python
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
```

After re-running the training command, Tensorboard file will be generated in the visualization folder `work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance/{timestamp}/vis_data`.
We can use Tensorboard to view the loss, learning rate, and coco/bbox_mAP visualizations from a web link by running the following command:

```shell
tensorboard --logdir=work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance
```

## Testing

```shell
python tools/test.py configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py \
                     work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance/best_coco_bbox_mAP_epoch_300.pth \
                     --show-dir show_results
```

Run the above test command, you can not only get the AP performance printed in the **Training** section, You can also automatically save the result images to the `work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance/{timestamp}/show_results` folder. Below is one of the result images, the left image is the actual annotation, and the right image is the inference result of the model.

<div align=center>
<img src="https://user-images.githubusercontent.com/87774050/236996421-46cfd38c-0827-4251-8216-408dfa9e03dd.jpg" alt="result_img"/>
</div>

You can also visualize model inference results in a browser window if you use `WandbVisBackend` or `TensorboardVisBackend`.

## Feature map visualization

MMYOLO provides visualization scripts for feature map to analyze the current model training. Please refer to [Feature Map Visualization](../recommended_topics/visualization.md)

Due to the bias of direct visualization of `test_pipeline`, we need to modify the `test_pipeline` of `configs/yolov5/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py`

```python
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
```

to the following config:

```python
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False), # modify the LetterResize to mmdet.Resize
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
```

Let's choose the `data/balloon/train/3927754171_9011487133_b.jpg` image as an example to visualize the output feature maps of YOLOv5 backbone and neck layers.

**1. Visualize the three channels of YOLOv5s backbone**

```shell
python demo/featmap_vis_demo.py data/balloon/train/3927754171_9011487133_b.jpg \
    configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py \
    work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance/best_coco_bbox_mAP_epoch_300.pth \ --target-layers backbone \
    --channel-reduction squeeze_mean
```

<div align=center>
<img src="https://user-images.githubusercontent.com/87774050/236997582-233e292f-5e96-4e44-9e92-9e0787f302fc.jpg" width="800" alt="image"/>
</div>

The result will be saved to the output folder in current path. Three output feature maps plotted in the above figure correspond to small, medium and large output feature maps.

**2. Visualize the three channels of YOLOv5 neck**

```shell
python demo/featmap_vis_demo.py data/balloon/train/3927754171_9011487133_b.jpg \
    configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py \
    work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance/best_coco_bbox_mAP_epoch_300.pth \ --target-layers neck \
    --channel-reduction squeeze_mean
```

<div align=center>
<img src="https://user-images.githubusercontent.com/87774050/236997860-719d2e18-7767-4129-a072-b21c97a5502a.jpg" width="800" alt="image"/>
</div>
**3. Grad-Based CAM visualization**

TODO

## EasyDeploy deployment

TODO

The full content above can be viewed in [15_minutes_object_detection.ipynb](../../../demo/15_minutes_object_detection.ipynb). This is the end of the tutorial. If you encounter problems during training or testing, please check the [common troubleshooting steps](../recommended_topics/troubleshooting_steps.md) first and feel free to open an [issue](https://github.com/open-mmlab/mmyolo/issues/new/choose) if you still can't solve it.
