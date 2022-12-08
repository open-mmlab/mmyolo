## YOLOv5使用其他的loss

如果想在 YOLOv5 中使用其他 `model` 中的 `loss` ，同时保持 YOLOv5 的网络结构，可以在 YOLOv5 的配置文件中修改。需要修改配置文件中 `model` 的 `bbox_head` 部分，将 `type` 修改为 YOLOv7Head 或 YOLOXHead ，而 `haed_module` 中的 `type` 则继续使用 `YOLOv5HeadModule`。其中，因为 YOLOX 是 Anchor Free 结构，所以需要将 `head_module` 中的 `num_base_priors` 设置为1，同时增加 `train_cfg` 中 `assigner` 的设置。

如果使用 YOLOv7Head 替代 YOLOv5Head，可以如下配置：

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

如果使用 YOLOXHead 替代 YOLOv5Head，可以如下配置：

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

另外，由于 YOLOv6 和 RTMDet 的 loss 中不包含 `loss_obj` ，所以如果希望在 YOLOv5 中使用对应的 YOLOv6Head 或 RTMDetHead，需要修改对应类中的 `loss_by_feat` 函数。
