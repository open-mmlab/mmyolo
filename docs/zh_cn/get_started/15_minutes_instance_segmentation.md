# 15 分钟上手 MMYOLO 实例分割

实例分割是计算机视觉中的一个任务，旨在将图像中的每个对象都分割出来，并为每个对象分配一个唯一的标识符。与语义分割不同，实例分割不仅分割出图像中的不同类别，还将同一类别的不同实例分开。

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/87774050/6fd6316f-d78d-48a5-a413-86e7a74583fd" alt="Instance Segmentation" width="100%"/>
</div>

以可供下载的气球 balloon 小数据集为例，带大家 15 分钟轻松上手 MMYOLO 实例分割。整个流程包含如下步骤：

- [环境安装](#环境安装)
- [数据集准备](#数据集准备)
- [配置准备](#配置准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [EasyDeploy 模型部署](#easydeploy-模型部署)

本文以 YOLOv5-s 为例，其余 YOLO 系列算法的气球 balloon 小数据集 demo 配置请查看对应的算法配置文件夹下。

## 环境安装

假设你已经提前安装好了 Conda，接下来安装 PyTorch

```shell
conda create -n mmyolo python=3.8 -y
conda activate mmyolo
# 如果你有 GPU
conda install pytorch torchvision -c pytorch
# 如果你是 CPU
# conda install pytorch torchvision cpuonly -c pytorch
```

安装 MMYOLO 和依赖库

```shell
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
pip install -U openmim
mim install -r requirements/mminstall.txt
# Install albumentations
mim install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

```{note}
温馨提醒：由于本仓库采用的是 OpenMMLab 2.0，请最好新建一个 conda 虚拟环境，防止和 OpenMMLab 1.0 已经安装的仓库冲突。
```

详细环境配置操作请查看 [安装和验证](./installation.md)

## 数据集准备

Balloon 数据集是一个包括 74 张图片的单类别数据集, 包括了训练所需的标注信息。 样例图片如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/87774050/236993643-f581b087-9231-48a5-810b-97a5f31abe63.png" alt="balloon dataset"/>
</div>

你只需执行如下命令即可下载并且直接用起来

```shell
python tools/misc/download_dataset.py --dataset-name balloon --save-dir ./data/balloon --unzip --delete
python ./tools/dataset_converters/balloon2coco.py
```

data 位于 mmyolo 工程目录下， `train.json`, `val.json` 中存放的是 COCO 格式的标注，`data/balloon/train`, `data/balloon/val` 中存放的是所有图片

## 配置准备

以 YOLOv5 算法为例，考虑到用户显存和内存有限，我们需要修改一些默认训练参数来让大家愉快的跑起来，核心需要修改的参数如下

- YOLOv5 是 Anchor-Based 类算法，不同的数据集需要自适应计算合适的 Anchor
- 默认配置是 8 卡，每张卡 batch size 为 16，现将其改成单卡，每张卡 batch size 为 4
- 原则上 batch size 改变后，学习率也需要进行线性缩放，但是实测发现不需要

具体操作为在 `configs/yolov5/ins_seg` 文件夹下新建 `yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py` 配置文件(为了方便大家直接使用，我们已经提供了该配置)，并把以下内容复制配置文件中。

```python
_base_ = './yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance.py'  # noqa

data_root = 'data/balloon/'
# 训练集标注路径
train_ann_file = 'train.json'
train_data_prefix = 'train/'  # 训练集图片路径
# 测试集标注路径
val_ann_file = 'val.json'
val_data_prefix = 'val/'  # 验证集图片路径
metainfo = {
    'classes': ('balloon', ), # 数据集类别名称
    'palette': [
        (220, 20, 60),
    ]
}
num_classes = 1
# 批处理大小batch size设置为 4
train_batch_size_per_gpu = 4
# dataloader 加载进程数
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

以上配置从 `yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance.py` 中继承，并根据 balloon 数据的特点更新了 `data_root`、`metainfo`、`train_dataloader`、`val_dataloader`、`num_classes` 等配置。

## 模型训练

```shell
python tools/train.py configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py
```

运行以上训练命令 `work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance` 文件夹会被自动生成，权重文件以及此次的训练配置文件将会保存在此文件夹中。 在 1660 低端显卡上，整个训练过程大概需要 30 分钟。

<div align=center>
<img src="https://user-images.githubusercontent.com/87774050/236995027-00a16a9e-2a2d-44cc-be8a-e2c8a36ff77f.png" alt="image"/>
</div>

在 `val.json` 上性能如下所示：

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

上述性能是通过 COCO API 打印，其中 -1 表示不存在对于尺度的物体。

### 一些注意事项

在训练过程中会打印如下关键警告：

- You are using `YOLOv5Head` with num_classes == 1. The loss_cls will be 0. This is a normal phenomenon.

这个警告都不会对性能有任何影响。第一个警告是说明由于当前训练的类别数是 1，根据 YOLOv5 算法的社区， 分类分支的 loss 始终是 0，这是正常现象。

### 中断后恢复训练

如果训练中途停止，可以在训练命令最后加上 `--resume` ,程序会自动从 `work_dirs` 中加载最新的权重文件恢复训练。

```shell
python tools/train.py configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py --resume
```

### 节省显存策略

上述配置大概需要 1.0G 显存，如果你的显存不够，可以考虑开启混合精度训练

```shell
python tools/train.py configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py --amp
```

### 训练可视化

MMYOLO 目前支持本地、TensorBoard 以及 WandB 等多种后端可视化，默认是采用本地可视化方式，你可以切换为 WandB 等实时可视化训练过程中各类指标。

#### 1 WandB 可视化使用

WandB 官网注册并在 https://wandb.ai/settings 获取到 WandB 的 API Keys。

<div align=center>
<img src="https://cdn.vansin.top/img/20220913212628.png" alt="image"/>
</div>

```shell
pip install wandb
# 运行了 wandb login 后输入上文中获取到的 API Keys ，便登录成功。
wandb login
```

在 `configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py` 配置文件最后添加 WandB 配置

```python
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
```

重新运行训练命令便可以在命令行中提示的网页链接中看到 loss、学习率和 coco/bbox_mAP 等数据可视化了。

```shell
python tools/train.py configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py
```

#### 2 Tensorboard 可视化使用

安装 Tensorboard 环境

```shell
pip install tensorboard
```

同上述在配置文件 `configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py`配置的最后添加 `tensorboard` 配置

```python
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
```

重新运行训练命令后，Tensorboard 文件会生成在可视化文件夹 `work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance/{timestamp}/vis_data` 下，
运行下面的命令便可以在网页链接使用 Tensorboard 查看 loss、学习率和 coco/bbox_mAP 等可视化数据了：

```shell
tensorboard --logdir=work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance
```

## 模型测试

```shell
python tools/test.py configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py \
                     work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance/best_coco_bbox_mAP_epoch_300.pth \
                     --show-dir show_results
```

运行以上测试命令， 你不仅可以得到**模型训练**部分所打印的 AP 性能，还可以将推理结果图片自动保存至 `work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance/{timestamp}/show_results` 文件夹中。下面为其中一张结果图片，左图为实际标注，右图为模型推理结果。

<div align=center>
<img src="https://user-images.githubusercontent.com/87774050/236996421-46cfd38c-0827-4251-8216-408dfa9e03dd.jpg" alt="result_img"/>
</div>

如果你使用了 `WandbVisBackend` 或者 `TensorboardVisBackend`，则还可以在浏览器窗口可视化模型推理结果。

## 特征图相关可视化

MMYOLO 中提供了特征图相关可视化脚本，用于分析当前模型训练效果。 详细使用流程请参考 [特征图可视化](../recommended_topics/visualization.md)

由于 `test_pipeline` 直接可视化会存在偏差，故将需要 `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py` 中 `test_pipeline`

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

修改为如下配置：

```python
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False), # 删除 YOLOv5KeepRatioResize, 将 LetterResize 修改成 mmdet.Resize
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))  # 删除 pad_param
]
```

我们选择 `data/balloon/train/3927754171_9011487133_b.jpg` 图片作为例子，可视化 YOLOv5 backbone 和 neck 层的输出特征图。

```shell
python demo/featmap_vis_demo.py data/balloon/train/3927754171_9011487133_b.jpg \
    configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py \
    work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance/best_coco_bbox_mAP_epoch_300.pth \ --target-layers backbone \
    --channel-reduction squeeze_mean
```

<div align=center>
<img src="https://user-images.githubusercontent.com/87774050/236997582-233e292f-5e96-4e44-9e92-9e0787f302fc.jpg" width="800" alt="image"/>
</div>

结果会保存到当前路径的 output 文件夹下。上图中绘制的 3 个输出特征图对应大中小输出特征图。

**2. 可视化 YOLOv5 neck 输出的 3 个通道**

```shell
python demo/featmap_vis_demo.py data/balloon/train/3927754171_9011487133_b.jpg \
    configs/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance.py \
    work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_balloon_instance/best_coco_bbox_mAP_epoch_300.pth \ --target-layers neck \
    --channel-reduction squeeze_mean
```

<div align=center>
<img src="https://user-images.githubusercontent.com/87774050/236997860-719d2e18-7767-4129-a072-b21c97a5502a.jpg" width="800" alt="image"/>
</div>

**3. Grad-Based CAM 可视化**

TODO

## EasyDeploy 模型部署

TODO

至此本教程结束。

以上完整内容可以查看 [15_minutes_instance_segmentation.ipynb](../../../demo/15_minutes_instance_segmentation.ipynb)。 如果你在训练或者测试过程中碰到问题，请先查看 [常见错误排除步骤](../recommended_topics/troubleshooting_steps.md)， 如果依然无法解决欢迎提 issue。
