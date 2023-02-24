# Plugins

MMYOLO supports adding plugins such as `none_local` and `dropblock` after different stages of Backbone. Users can directly manage plugins by modifying the plugins parameter of the backbone in the config. For example, add `GeneralizedAttention` plugins for `YOLOv5`. The configuration files are as follows:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type='GeneralizedAttention',
                    spatial_range=-1,
                    num_heads=8,
                    attention_type='0011',
                    kv_stride=2),
                stages=(False, False, True, True))
        ]))
```

`cfg` parameter indicates the specific configuration of the plugin. The `stages` parameter indicates whether to add plug-ins after the corresponding stage of the backbone. The length of the list `stages` must be the same as the number of backbone stages.

MMYOLO currently supports the following plugins:

<details open>
<summary><b>Supported Plugins</b></summary>

1. [CBAM](https://github.com/open-mmlab/mmyolo/blob/dev/mmyolo/models/plugins/cbam.py#L86)
2. [GeneralizedAttention](https://github.com/open-mmlab/mmcv/blob/2.x/mmcv/cnn/bricks/generalized_attention.py#L13)
3. [NonLocal2d](https://github.com/open-mmlab/mmcv/blob/2.x/mmcv/cnn/bricks/non_local.py#L250)
4. [ContextBlock](https://github.com/open-mmlab/mmcv/blob/2.x/mmcv/cnn/bricks/context_block.py#L18)

</details>
