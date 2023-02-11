# 旋转目标检测

所谓旋转目标检测（Rotated Object Detection），又称为有向目标检测（Oriented Object Detection），试图在检测出目标位置的同时得到目标的方向信息。它通过重新定义目标表示形式，以及增加回归自由度数量的操作，实现旋转矩形、四边形甚至任意形状的目标检测。旋转目标检测在人脸识别、场景文字、遥感影像、自动驾驶、医学图像、机器人抓取等领域都有广泛应用。

关于旋转目标检测的详细介绍请参考文档 [MMRotate 基础知识](https://mmrotate.readthedocs.io/zh_CN/1.x/overview.html)

MMYOLO 中的旋转目标检测依赖于MMRotate 1.0，请参考文档 [开始你的第一步](https://mmrotate.readthedocs.io/zh_CN/1.x/get_started.html) 安装MMRotate 1.0。

本教程将介绍如何在 MMYOLO 中训练和使用旋转目标检测模型，目前支持了RTMDet-R。

## 数据集准备

对于旋转目标检测数据集，目前最常用的数据集是DOTA数据集，由于DOTA数据集中的图像分辨率较大，因此需要进行切片处理，数据集准备请参考 [Preparing DOTA Dataset](https://github.com/open-mmlab/mmrotate/blob/1.x/tools/data/dota/README.md).

对于自定义数据集，我们建议将数据转换为 DOTA 格式并离线进行转换，如此您只需在数据转换后修改 config 的数据标注路径和类别即可。

为了方便使用，我们同样提供了基于 COCO 格式的旋转标注格式，将多边形检测框储存在COCO标注的segmentation标签中，示例如下：

```json
{
    "id": 131,
    "image_id": 72,
    "bbox": [123, 167, 11, 37],
    "area": 271.5,
    "category_id": 1,
    "segmentation": [[123, 167, 128, 204, 134, 201, 132, 167]],
    "iscrowd": 0,
}
```

## 配置文件

这里以RTMDet-R为例介绍旋转目标检测的配置文件，其中大部分和水平检测模型相同，主要介绍它们的差异，包括数据集和评测器配置、检测头、可视化等。

得益于 MMEngine 的配置文件系统，大部分模块都可以调用MMRotate中的模块。

### 数据集和评测器配置

关于配置文件的基础请先阅读 [学习 YOLOV5 配置文件](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/config.html). 下面介绍旋转目标检测的一些必要设置。

```python
dataset_type = 'YOLOv5DOTADataset'  # 数据集类型，这将被用来定义数据集
data_root = 'data/split_ss_dota/'  # 数据的根路径
file_client_args = dict(backend='disk')  # 文件读取后端的配置，默认从硬盘读取

angle_version = 'le90' # 角度范围的定义

train_pipeline = [
    # 训练数据读取流程
    dict(
        type='LoadImageFromFile', # 第 1 个流程，从文件路径里加载图像
        file_client_args=file_client_args),  # 文件读取后端的配置，默认从硬盘读取
    dict(type='LoadAnnotations', # 第 2 个流程，对于当前图像，加载它的注释信息
         with_bbox=True, # 是否使用标注框(bounding box)，目标检测需要设置为 True
         box_type='qbox'), # 指定读取的标注格式，旋转框数据集默认的数据格式为四边形
    dict(type='mmrotate.ConvertBoxType', # 第 3 个流程，转换标注格式
         box_type_mapping=dict(gt_bboxes='rbox')), # 将四边形标注转化为旋转框标注

    # 训练数据处理流程
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.RandomFlip',
         prob=0.75,
         direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmrotate.RandomRotate', # 旋转数据增强
         prob=0.5, # 旋转概率 0.5
         angle_range=180, # 旋转范围 180
         rotate_type='mmrotate.Rotate', # 旋转方法
         rect_obj_labels=[9, 11]), # 由于DOTA数据集中标号为9的 'storage-tank' 和标号11的 'roundabout' 两类为正方形标注，无需角度信息，旋转中将这两类保持为水平
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='RegularizeRotatedBox', # 统一旋转框表示形式
         angle_version=angle_version), # 根据角度的定义方式进行
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict( # 训练数据集的配置
        type=dataset_type,
        data_root=data_root,
        ann_file='trainval/annfiles/', # 标注文件夹路径
        data_prefix=dict(img_path='trainval/images/'), # 图像路径前缀
        img_shape=(1024, 1024), # 图像大小
        filter_cfg=dict(filter_empty_gt=True), # 标注的过滤配置
        pipeline=train_pipeline)) # 这是由之前创建的 train_pipeline 定义的数据处理流程
```

RTMDet-R 保持论文内的配置，默认仅采用随机旋转增强，得益于BoxType设计，在数据增强阶段，大部分增强无需改动代码即可直接支持，例如MixUp和Mosaic等，可以直接在pipeline中使用。

```{Warning}
目前已知Albu数据增强仅支持水平框，在使用其他的数据增强时建议先使用 可视化数据集脚本 `browse_dataset.py` 验证数据增强是否正确。
```

RTMDet-R 测试阶段仅采用Resize和Pad，在验证和评测时，都采用相同的数据流进行推理。

```python
val_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    # 和训练时一致，先读取标注再转换标注格式
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        box_type='qbox',
        _scope_='mmdet'),
    dict(
        type='mmrotate.ConvertBoxType',
        box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=val_pipeline))

test_dataloader = val_dataloader
```

[评测器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/evaluation.html) 用于计算训练模型在验证和测试数据集上的指标。评测器的配置由一个或一组评价指标（Metric）配置组成：

```python
val_evaluator = dict( # 验证过程使用的评测器
    type='mmrotate.DOTAMetric', # 用于评估旋转目标检测的 mAP 的 dota 评价指标
    metric='mAP' # 需要计算的评价指标
)
test_evaluator = val_evaluator  # 测试过程使用的评测器
```

由于DOTA测试数据集没有标注文件， 如果要保存在测试数据集上的检测结果，则可以像这样编写配置：

```python
# 在测试集上推理，
# 并将检测结果转换格式以用于提交结果
test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='mmrotate.DOTAMetric',
    format_only=True, # 只将模型输出转换为DOTA的txt提交格式并压缩成zip
    merge_patches=True, # 将切片结果合并成大图检测结果
    outfile_prefix='./work_dirs/dota_detection/submission') # 输出测试文件夹的路径
```

如果使用基于COCO格式的旋转框标注，只需要修改pipeline中数据读取流程和训练数据集的配置，以训练数据为例：

```python

dataset_type='YOLOv5CocoDataset'

train_pipeline = [
    # 训练数据读取流程
    dict(
        type='LoadImageFromFile', # 第 1 个流程，从文件路径里加载图像
        file_client_args=file_client_args),  # 文件读取后端的配置，默认从硬盘读取
    dict(type='LoadAnnotations', # 第 2 个流程，对于当前图像，加载它的注释信息
         with_bbox=True, # 是否使用标注框(bounding box)，目标检测需要设置为 True
         with_mask=True, # 读取储存在mask标注中的多边形标注
         poly2mask=False) # 不执行poly2mask，后续会将poly转化成检测框
    dict(type='ConvertMask2BoxType', # 第 3 个流程，将mask标注转化为 boxtype
         box_type='rbox'), # 目标类型是 rbox 旋转框
    # 剩余的其他pipeline
    ...
]

metainfo = dict( # DOTA数据集的metainfo
    classes=('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
             'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
             'basketball-court', 'storage-tank', 'soccer-ball-field',
             'roundabout', 'harbor', 'swimming-pool', 'helicopter'))

train_dataloader = dict(
    dataset=dict( # 训练数据集的配置
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train/train.json', # 标注文件路径
        data_prefix=dict(img='train/images/'), # 图像路径前缀
        filter_cfg=dict(filter_empty_gt=True), # 标注的过滤配置
        pipeline=train_pipeline), # 数据处理流程
)
```

### 模型配置

对于旋转目标检测器，在模型配置中 backbone 和 neck 的配置和其他模型是一致的，主要差异在检测头上。

目前仅支持RTMDet-R旋转目标检测器。

### 可视化器

由于旋转框和水平框的差异，旋转目标检测模型需要使用MMRotate中的 `RotLocalVisualizer`，配置如下：

```python
vis_backends = [dict(type='LocalVisBackend')]  # 可视化后端，请参考 https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html
visualizer = dict(
    type='mmrotate.RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

## 实用工具

目前测试可用的工具包括：

[可视化数据集](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#id3)

<!-- [可视化数据集分析 TODO](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#id3)  -->
