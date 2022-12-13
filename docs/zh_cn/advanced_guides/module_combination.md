## model 和 loss 组合替换

在 MMYOLO 中，model 即网络本身和 loss 是解耦的，用户可以简单的通过修改配置文件中 model 和 loss 来组合不同模块。下面给出两个具体例子。

(1) YOLOv5 model 组合 YOLOv7 loss

```python
model = dict(
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
            featmap_strides=strides,
            num_base_priors=3),
        )
)
```

(2) YOLOv5 model 组合 YOLOX loss

```python
model = dict(
    bbox_head=dict(
        type='YOLOXHead',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
            featmap_strides=strides,
            num_base_priors=1)),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.SimOTAAssigner',
            center_radius=2.5,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'))),
)
```
