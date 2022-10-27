This tutorial collects answers to any `How to xxx with MMYOLO`. Feel free to update this doc if you meet new questions about `How to` and find the answers!

# Add plugins to the BackBone network

MMYOLO supports adding plug-ins such as none_local and dropout after different stages of BackBone. Users can directly manage plug-ins by modifying the plugins parameter of backbone in config. For example, add GeneralizedAttention plug-ins for `YOLOv5`. The configuration files are as follows:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type='mmdet.GeneralizedAttention',
                    spatial_range=-1,
                    num_heads=8,
                    attention_type='0011',
                    kv_stride=2),
                stages=(False, False, True, True)),
        ], ))
```

`cfg` parameter indicates the specific configuration of the plug-in. The `stages` parameter indicates whether to add plug-ins after the corresponding stage of the backbone. The length of list `stages` must be the same as the number of backbone stages.

## Apply multiple Necks

If you want to stack multiple Necks, you can directly set the Neck parameters in the config. MMYOLO supports concatenating multiple Necks in the form of `List`. You need to ensure that the output channel of the previous Neck matches the input channel of the next Neck. If you need to adjust the number of channels, you can insert the `mmdet.ChannelMapper` module to align the number of channels between multiple Necks. The specific configuration is as follows:

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
            out_channels=[256, 512, 1024], # The out_channels is controlled by widen_factor，so the YOLOv5PAFPN's out_channels equls to out_channels * widen_factor
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
    ]
    bbox_head=dict(head_module=dict(in_channels=[512,512,512])) # The out_channels is controlled by widen_factor，so the YOLOv5HeadModuled in_channels * widen_factor equals to  the last neck's out_channels
)
```
