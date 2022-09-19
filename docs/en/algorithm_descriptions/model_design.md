# Model design related instructions

## YOLO series model basic class

The structural Graph is provided by @RangeKing ,Thank you RangeKing！
![base class](https://user-images.githubusercontent.com/33799979/190382319-6b4e1fcb-cc3f-4fbe-9d6b-3c9c4e57472c.png)

Most of the YOLO series algorithms adopt a unified algorithm building structure, typically as Darknet + PAFPN. In order to let users quickly understand the YOLO series algorithm architecture, we deliberately designed the BaseBackbone + BaseYOLONeck structure as shown in the above graph.

The benefit of abstract BaseBackbone includes:

1. Subclasses do not need to concern about the forward process, just build the model as the builder pattern.
2. It can be configured to achieve custom plug-in functions, the users can easily insert some similar attention module.
3. All subclasses automatically support frozen certain stage and frozen bn functions.

BaseYOLONeck has the same benefit as BaseBackbone.

### BaseBackbone

We can see in the above graph，as for P5，BaseBackbone include 1 stem layer and 4 stage layers which are similar to the basic structural of ResNet. Different backbone network algorithms inheritance the BaseBackbone, users can achieve construction of every layer of the network by using self-custom basic module  through  `build_xx` method.

### BaseYOLONeck

We reproduce the YOLO series Neck component by the similar method of the BaseBackbone, we can mainly divide them into Reduce layer, UpSample layer,TopDown layer,DownSample layer，BottomUP layer and output convolution layer, every layer can self-custom its inside construction by inheritance and rewrite `build_xx` method.

### BaseDenseHead

The YOLO series uses the BaseDenseHead designed in MMDet as the base class of the Head structure. Take YOLOv5 as an example, [HeadModule](https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/models/dense_heads/yolov5_head.py#L2) class's forward function replace original forward method.

## HeadModule

<div align=center>
<img src="https://user-images.githubusercontent.com/33799979/190407754-c725fe85-a71b-4e45-912b-34513d1ff128.png" width=800>
</div>

Methods implementation in the [MMDetection](https://github.com/open-mmlab/mmdetection) is shown in the above graph ，The solid line is the implementation in [MMYOLO](https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/models/dense_heads/yolov5_head.py), which has the following advantages over the original implementation:

1. MMDet in the bbox_head split into assigner + box coder + sampler three large components, but for the generality of passing through the 3 components , the model need to encapsulate additional objects to handle, and after the unification, the user needn't separate them. The benefits of not deliberately forcing the division of the three components are: no longer need to data encapsulation of internal data, simplifying the code logic, reducing the difficulty of use and the difficulty of algorithm implementation.
2. MMYOLO is Faster, the user can customize the implementation of the algorithm when the original framework does not depend on the deep optimization of part of the code.

In general, in the MMYOLO, they only need to implement the decouple of the model + loss_by_feat parts, and users can achieve any model with any `loss_by_feat` calculation process through modify the configuration. For example, applying the YOLOX `loss_by_feat` to the YOLOv5 model, etc.

Taking the YOLOX configuration in MMDet as an example, the Head module configuration is written as follows:

```python
bbox_head=dict(
    type='YOLOXHead',
    num_classes=80,
    in_channels=128,
    feat_channels=128,
    stacked_convs=2,
    strides=(8, 16, 32),
    use_depthwise=False,
    norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
    act_cfg=dict(type='Swish'),
    ...
    loss_obj=dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='sum',
        loss_weight=1.0),
    loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
```

After extracting the head_module in MMYOLO, the new configuration is written as follows:

```python
bbox_head=dict(
    type='YOLOXHead',
    head_module=dict(
        type='YOLOXHeadModule',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        widen_factor=widen_factor,
        stacked_convs=2,
        featmap_strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    ...
    loss_obj=dict(
        type='mmdet.CrossEntropyLoss',
        use_sigmoid=True,
        reduction='sum',
        loss_weight=1.0),
    loss_bbox_aux=dict(type='mmdet.L1Loss', reduction='sum', loss_weight=1.0)),
train_cfg=dict(
    assigner=dict(
        type='mmdet.SimOTAAssigner',
        center_radius=2.5,
        iou_calculator=dict(type='mmdet.BboxOverlaps2D'))),
```
