# 给主干网络增加插件

MMYOLO 支持在 Backbone 的不同 Stage 后增加如 `none_local`、`dropblock` 等插件，用户可以直接通过修改 config 文件中 `backbone` 的 `plugins`参数来实现对插件的管理。例如为 `YOLOv5` 增加`GeneralizedAttention` 插件，其配置文件如下：

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

`cfg` 参数表示插件的具体配置， `stages` 参数表示是否在 backbone 对应的 stage 后面增加插件，长度需要和 backbone 的 stage 数量相同。

目前 `MMYOLO` 支持了如下插件：

<details open>
<summary><b>支持的插件</b></summary>

1. [CBAM](https://github.com/open-mmlab/mmyolo/blob/dev/mmyolo/models/plugins/cbam.py#L86)
2. [GeneralizedAttention](https://github.com/open-mmlab/mmcv/blob/2.x/mmcv/cnn/bricks/generalized_attention.py#L13)
3. [NonLocal2d](https://github.com/open-mmlab/mmcv/blob/2.x/mmcv/cnn/bricks/non_local.py#L250)
4. [ContextBlock](https://github.com/open-mmlab/mmcv/blob/2.x/mmcv/cnn/bricks/context_block.py#L18)

</details>
