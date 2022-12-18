# Backbone + Neck + Head 的组合替换

## 1. YOLOv5 Backbone 替换

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

## 2. YOLOv5 Neck 替换

(1) 假设想将 `yolov5 backbone + yolov6 neck + yolov5 head` 作为 `YOLOv5` 的完整网络，则配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    neck = dict(
        type = 'YOLOv6RepPAFPN',
        in_channels = [256, 512, 1024],
        out_channels = [128, 256, 512], # 注意YOLOv6RepPAFPN的输出通道是[128, 256, 512]
        num_csp_blocks = 12,
        act_cfg = dict(type='ReLU', inplace = True),
    ),
    bbox_head = dict(
        head_module = dict(
            in_channels = [128, 256, 512])) # head部分输入通道要做相应更改
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
            in_channels = [256, 512, 1024])) # 注意使用YOLOv7PAFPN后head部分输入通道数是neck输出通道数的两倍
)
```

## 3. YOLOv5 Head 替换

(1) 假设想将 `yolov5 backbone + yolov5 neck + yolo7 head` 作为 `YOLOv5` 的完整网络，则配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# different from yolov5
anchors = [
    [(12, 16), (19, 36), (40, 28)],  # P3/8
    [(36, 75), (76, 55), (72, 146)],  # P4/16
    [(142, 110), (192, 243), (459, 401)]  # P5/32
]
strides = [8, 16, 32]
num_det_layers = 3
num_classes = 1 # 根据自己的数据集调整
img_scale = (640, 640)

model = dict(
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            featmap_strides=strides,
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.3 * (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='mean',
            loss_weight=0.05 * (3 / num_det_layers),
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.7 * ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
        obj_level_weights=[4., 1., 0.4],
        # BatchYOLOv7Assigner params
        prior_match_thr=4.,
        simota_candidate_topk=10,
        simota_iou_weight=3.0,
        simota_cls_weight=1.0))
```
