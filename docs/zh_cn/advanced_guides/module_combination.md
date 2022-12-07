# 算法组合替换教程

## Loss 组合替换教程

OpenMMLab 2.0 体系中 MMYOLO、MMDetection、MMClassification 中的 loss 注册表都继承自 MMEngine 中的根注册表。 因此用户可以在 MMYOLO 中使用来自 MMDetection、MMClassification 中实现的 loss 而无需重新实现。

### 替换 YOLOv5 Head 中的 loss_cls 函数

1. 假设我们想使用 `LabelSmoothLoss` 作为 `loss_cls` 的损失函数。因为 `LabelSmoothLoss` 已经在 MMClassification 中实现了，所以可以直接在配置文件中进行替换。配置文件如下：

```python
# 请先使用命令： mim install "mmcls>=1.0.0rc2"，安装 mmcls
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model = dict(
    bbox_head=dict(
      loss_cls=dict(
        _delete_=True,
        _scope_='mmcls', #  临时替换 scope 为 mmcls
        type='LabelSmoothLoss',
        label_smooth_val=0.1,
        mode='multi_label',
        reduction='mean',
        loss_weight=0.5)))
```

2. 假设我们想使用 `VarifocalLoss` 作为 `loss_cls` 的损失函数。因为 `VarifocalLoss` 在 MMDetection 已经实现好了，所以可以直接替换。配置文件如下：

```python
model = dict(
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            _scope_='mmdet',
            type='VarifocalLoss',
            loss_weight=1.0)))
```

3. 假设我们想使用 `FocalLoss` 作为 `loss_cls` 的损失函数。配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model = dict(
    bbox_head=dict(
        loss_cls= dict(
            _delete_=True,
            type='FocalLoss',
            loss_weight=1.0)))
```

4. 假设我们想使用 `QualityFocalLoss` 作为 `loss_cls` 的损失函数。配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model = dict(
    bbox_head=dict(
      loss_cls= dict(
        _delete_=True,
        type='QualityFocalLoss',
        loss_weight=1.0)))
```

### 替换 YOLOV5 Head 中的 loss_obj 函数

`loss_obj` 的替换与 `loss_cls` 的替换类似，我们可以使用已经实现好的损失函数对 `loss_obj` 的损失函数进行替换

1. 假设我们想使用 `VarifocalLoss` 作为 `loss_obj` 的损失函数

```python
model = dict(
    bbox_head=dict(
        loss_obj=dict(
            _delete_=True,
            _scope_='mmdet',
            type='VarifocalLoss',
            loss_weight=1.0)))
```

2. 假设我们想使用 `FocalLoss` 作为 `loss_obj` 的损失函数。

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model = dict(
    bbox_head=dict(
        loss_cls= dict(
            _delete_=True,
            type='FocalLoss',
            loss_weight=1.0)))
```

3. 假设我们想使用 `QualityFocalLoss` 作为 `loss_obj` 的损失函数。

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model = dict(
    bbox_head=dict(
      loss_cls= dict(
        _delete_=True,
        type='QualityFocalLoss',
        loss_weight=1.0)))
```

#### 注意

1. 在本教程中损失函数的替换是运行不报错的，但无法保证性能一定会上升。
2. 本次损失函数的替换都是以 YOLOv5 算法作为例子的，但是 MMYOLO 下的多个算法，如 YOLOv6，YOLOX 等算法都可以按照上述的例子进行替换。
