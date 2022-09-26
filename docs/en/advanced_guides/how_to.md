This tutorial collects answers to any `How to xxx with MMYOLO`. Feel free to update this doc if you meet new questions about `How to` and find the answers!

# Add plugins to the BackBone network

MMYOLO supports adding plug-ins such as none_local and dropout after different stages of BackBone. Users can directly manage plug-ins by modifying the plugins parameter of backbone in config. For example, add DropBlock and GeneralizedAttention plug-ins for `YOLOv5`. The configuration files are as follows:

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

`cfg`parameter indicates the specific configuration of the plug-in. The `stages` parameter indicates whether to add plug-ins after the corresponding stage of the backbone. The length of list `stages` must be the same as the number of backbone stages.
