# 应用多个 Neck

如果你想堆叠多个 Neck，可以直接在配置文件中的 Neck 参数，MMYOLO 支持以 `List` 形式拼接多个 Neck 配置，你需要保证上一个 Neck 的输出通道与下一个 Neck
的输入通道相匹配。如需要调整通道，可以插入 `mmdet.ChannelMapper` 模块用来对齐多个 Neck 之间的通道数量。具体配置如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

deepen_factor = _base_.deepen_factor
widen_factor = _base_.widen_factor
model = dict(
    type='YOLODetector',
    neck=[
        dict(
            type='YOLOv5PAFPN',
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            in_channels=[256, 512, 1024],
            out_channels=[256, 512, 1024],
            # 因为 out_channels 由 widen_factor 控制，YOLOv5PAFPN 的 out_channels = out_channels * widen_factor
            num_csp_blocks=3,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        dict(
            type='mmdet.ChannelMapper',
            in_channels=[128, 256, 512],
            out_channels=128,
        ),
        dict(
            type='mmdet.DyHead',
            in_channels=128,
            out_channels=256,
            num_blocks=2,
            # disable zero_init_offset to follow official implementation
            zero_init_offset=False)
    ],
    bbox_head=dict(head_module=dict(in_channels=[512, 512, 512]))
    # 因为 out_channels 由 widen_factor 控制，YOLOv5HeadModuled 的 in_channels * widen_factor 才会等于最后一个 neck 的 out_channels
)
```
