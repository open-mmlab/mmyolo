# Freeze layers

## Freeze the weight of backbone

In MMYOLO, we can freeze some `stages` of the backbone network by setting `frozen_stages` parameters, so that these `stage` parameters do not participate in model updating.
It should be noted that `frozen_stages = i` means that all parameters from the initial `stage` to the `i`<sup>th</sup> `stage` will be frozen. The following is an example of `YOLOv5`. Other algorithms are the same logic.

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    backbone=dict(
        frozen_stages=1 # Indicates that the parameters in the first stage and all stages before it are frozen
    ))
```

## Freeze the weight of neck

In addition, it's able to freeze the whole `neck` with the parameter `freeze_all` in MMYOLO. The following is an example of `YOLOv5`. Other algorithms are the same logic.

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    neck=dict(
        freeze_all=True # If freeze_all=True, all parameters of the neck will be frozen
    ))
```
