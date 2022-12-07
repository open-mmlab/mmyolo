## YOLOV5使用其他的loss

如果想在YOLOV5中使用其他model中的loss，同时保持YOLOV5的网络结构，可以在YOLOV5的配置文件中修改。需要修改配置文件中model的bbox_head部分，将type修改未YOLOv7Head或YOLOXHead，而haed_module中的type则继续使用YOLOv5HeadModule。其中，因为YOLOX是Anchor Free结构，所以需要将head_module中的num_base_priors设置为，同时增加train_cfg中assigner的设置。

如果使用YOLOv7Head替代YOLOv5Head，可以如下配置：

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

如果使用YOLOXHead替代YOLOv5Head，可以如下配置：

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

另外，由于YOLOv6和RTMDet的loss中不包含loss_obj，所以如果希望在YOLOv5中使用对应的YOLOv6Head或RTMDetHead，需要修改对应类中的loss_by_feat函数。
