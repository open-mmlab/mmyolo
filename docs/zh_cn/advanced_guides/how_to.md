本教程收集了任何如何使用 MMYOLO 进行 xxx 的答案。 如果您遇到有关`如何做`的问题及答案，请随时更新此文档！

## 给骨干网络增加插件

MMYOLO支持在BackBone的不同Stage后增加如none_local、dropout等插件，用户可以直接通过修改config中backbone的plugins参数来实现对插件的管理。例如为`YOLOv5`增加DropBlock和GeneralizedAttention插件，其配置文件如下：

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

`cfg`参数表示插件的具体配置，`stages`参数表示是否在backbone对应的stage后面增加插件，长度需要和backbone的stage数量相同。
