# 冻结指定网络层权重

## 冻结 backbone 权重

在 MMYOLO 中我们可以通过设置 `frozen_stages` 参数去冻结主干网络的部分 `stage`, 使这些 `stage` 的参数不参与模型的更新。
需要注意的是：`frozen_stages = i` 表示的意思是指从最开始的 `stage` 开始到第 `i` 层 `stage` 的所有参数都会被冻结。下面是 `YOLOv5` 的例子，其他算法也是同样的逻辑：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    backbone=dict(
        frozen_stages=1 # 表示第一层 stage 以及它之前的所有 stage 中的参数都会被冻结
    ))
```

## 冻结 neck 权重

MMYOLO 中也可以通过参数 `freeze_all` 去冻结整个 `neck` 的参数。下面是 `YOLOv5` 的例子，其他算法也是同样的逻辑：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    neck=dict(
        freeze_all=True # freeze_all=True 时表示整个 neck 的参数都会被冻结
    ))
```
