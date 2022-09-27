本教程收集了任何如何使用 MMYOLO 进行 xxx 的答案。 如果您遇到有关`如何做`的问题及答案，请随时更新此文档！

## 给骨干网络增加插件

MMYOLO支持在BackBone的不同Stage后增加如none_local、dropout等插件，用户可以直接通过修改配置文件中backbone的plugins参数来实现对插件的管理。例如为`YOLOv5`增加DropBlock和GeneralizedAttention插件，其配置文件如下：

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

`cfg`参数表示插件的具体配置，`stages`参数表示是否在backbone对应的stage后面增加插件，列表`stages`的长度需要和backbone的stage数量相同。

## 应用多个Neck

如果你想堆叠多个Neck，可以直接在配置文件中的Neck参数，MMYOLO支持以`List`形式拼接多个Neck配置，你需要保证的是上一个Neck的输出通道与下一个Neck的输入通道相匹配。如需要调整通道，可以插入`mmdet.ChannelMapper`模块用来对齐多个Neck之间的通道数量。具体配置如下：

```python
model = dict(
    type='YOLODetector',
    neck=[
        dict(
            type='YOLOv5PAFPN',
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            in_channels=[256, 512, 1024],
            out_channels=[256, 512, 1024],
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
```
