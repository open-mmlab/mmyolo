# 算法组合替换教程

## Loss 组合替换教程

OpenMMLab 2.0 体系中 MMYOLO、MMDetection、MMClassification 中的 loss 注册表都继承自 MMEngine 中的根注册表。 因此用户可以在 MMYOLO 中使用来自 MMDetection、MMClassification 中实现的 loss 而无需重新实现。

### 替换 YOLOv5 Head 中的 loss_cls 函数

1. 假设我们想使用 `LabelSmoothLoss` 作为 `loss_cls` 的损失函数。因为 `LabelSmoothLoss` 已经在 MMClassification 中实现了，所以可以直接在配置文件中进行替换。配置文件如下：

```python
# 请先使用命令： mim install "mmcls>=1.0.0rc2"，安装 mmcls
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model = dict(
    bbox_head=dict(
      loss_cls=dict(
        _delete_=True,
        _scope_='mmcls', #  临时替换 scope 为 mmcls
        type='LabelSmoothLoss',
        label_smooth_val=0.1,
        mode='multi_label',
        reduction='mean',
        loss_weight=0.5)))
```

2. 假设我们想使用 `VarifocalLoss` 作为 `loss_cls` 的损失函数。因为 `VarifocalLoss` 在 MMDetection 已经实现好了，所以可以直接替换。配置文件如下：

```python
model = dict(
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            _scope_='mmdet',
            type='VarifocalLoss',
            loss_weight=1.0)))
```

3. 假设我们想使用 `FocalLoss` 作为 `loss_cls` 的损失函数。配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model = dict(
    bbox_head=dict(
        loss_cls= dict(
            _delete_=True,
            _scope_='mmdet',
            type='FocalLoss',
            loss_weight=1.0)))
```

4. 假设我们想使用 `QualityFocalLoss` 作为 `loss_cls` 的损失函数。配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model = dict(
    bbox_head=dict(
      loss_cls= dict(
        _delete_=True,
        _scope_='mmdet',
        type='QualityFocalLoss',
        loss_weight=1.0)))
```

### 替换 YOLOv5 Head 中的 loss_obj 函数

`loss_obj` 的替换与 `loss_cls` 的替换类似，我们可以使用已经实现好的损失函数对 `loss_obj` 的损失函数进行替换

1. 假设我们想使用 `VarifocalLoss` 作为 `loss_obj` 的损失函数

```python
model = dict(
    bbox_head=dict(
        loss_obj=dict(
            _delete_=True,
            _scope_='mmdet',
            type='VarifocalLoss',
            loss_weight=1.0)))
```

2. 假设我们想使用 `FocalLoss` 作为 `loss_obj` 的损失函数。

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model = dict(
    bbox_head=dict(
        loss_cls= dict(
            _delete_=True,
            _scope_='mmdet',
            type='FocalLoss',
            loss_weight=1.0)))
```

3. 假设我们想使用 `QualityFocalLoss` 作为 `loss_obj` 的损失函数。

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model = dict(
    bbox_head=dict(
      loss_cls= dict(
        _delete_=True,
        _scope_='mmdet',
        type='QualityFocalLoss',
        loss_weight=1.0)))
```

#### 注意

1. 在本教程中损失函数的替换是运行不报错的，但无法保证性能一定会上升。
2. 本次损失函数的替换都是以 YOLOv5 算法作为例子的，但是 MMYOLO 下的多个算法，如 YOLOv6，YOLOX 等算法都可以按照上述的例子进行替换。

## Model 和 Loss 组合替换

在 MMYOLO 中，model 即网络本身和 loss 是解耦的，用户可以简单的通过修改配置文件中 model 和 loss 来组合不同模块。下面给出两个具体例子。

(1) YOLOv5 model 组合 YOLOv7 loss，配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model = dict(
    bbox_head=dict(
        _delete_=True,
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=80,
            in_channels=[256, 512, 1024],
            widen_factor=0.5,
            featmap_strides=[8, 16, 32],
            num_base_priors=3)))
```

(2) RTMDet model 组合 YOLOv6 loss，配置文件如下：

```python
_base_ = './rtmdet_l_syncbn_8xb32-300e_coco.py'
model = dict(
    bbox_head=dict(
        _delete_=True,
        type='YOLOv6Head',
        head_module=dict(
            type='RTMDetSepBNHeadModule',
            num_classes=80,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU', inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=[8, 16, 32]),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='giou',
            bbox_format='xyxy',
            reduction='mean',
            loss_weight=2.5,
            return_iou=False)),
    train_cfg=dict(
        _delete_=True,
        initial_epoch=4,
        initial_assigner=dict(
            type='BatchATSSAssigner',
            num_classes=80,
            topk=9,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=80,
            topk=13,
            alpha=1,
            beta=6)
    ))
```

## Backbone + Neck + HeadModule 的组合替换

### 1. YOLOv5 Backbone 替换

(1) 假设想将 `RTMDet backbone + yolov5 neck + yolov5 head` 作为 `YOLOv5` 的完整网络，则配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

widen_factor = 0.5
deepen_factor = 0.33

model = dict(
    backbone=dict(
        _delete_=True,
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True))
)
```

(2)  `YOLOv6EfficientRep backbone + yolov5 neck + yolov5 head` 作为 `YOLOv5` 的完整网络，则配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    backbone=dict(
        type='YOLOv6EfficientRep',
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True))
)
```

### 2. YOLOv5 Neck 替换

(1) 假设想将 `yolov5 backbone + yolov6 neck + yolov5 head` 作为 `YOLOv5` 的完整网络，则配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    neck = dict(
        type = 'YOLOv6RepPAFPN',
        in_channels = [256, 512, 1024],
        out_channels = [128, 256, 512], # 注意 YOLOv6RepPAFPN 的输出通道是[128, 256, 512]
        num_csp_blocks = 12,
        act_cfg = dict(type='ReLU', inplace = True),
    ),
    bbox_head = dict(
        head_module = dict(
            in_channels = [128, 256, 512])) # head 部分输入通道要做相应更改
)
```

(2) 假设想将 `yolov5 backbone + yolov7 neck + yolov5 head` 作为 `YOLOv5` 的完整网络，则配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

deepen_factor = _base_.deepen_factor
widen_factor = _base_.widen_factor

model = dict(
    neck = dict(
        _delete_=True, # 将 _base_ 中关于 neck 的字段删除
        type = 'YOLOv7PAFPN',
        deepen_factor = deepen_factor,
        widen_factor = widen_factor,
        upsample_feats_cat_first = False,
        in_channels = [256, 512, 1024],
        out_channels = [128, 256, 512],
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg = dict(type='SiLU', inplace=True),
    ),
    bbox_head = dict(
        head_module = dict(
            in_channels = [256, 512, 1024])) # 注意使用 YOLOv7PAFPN 后 head 部分输入通道数是 neck 输出通道数的两倍
)
```

### 3. YOLOv5 HeadModule 替换

(1) 假设想将 `yolov5 backbone + yolov5 neck + yolo7 headmodule` 作为 `YOLOv5` 的完整网络，则配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

strides = [8, 16, 32]
num_classes = 1 # 根据自己的数据集调整

model = dict(
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            featmap_strides=strides,
            num_base_priors=3)))
```
