# TTA Related Notes

## Test Time Augmentation (TTA)

MMYOLO support for TTA in v0.5.0+, so that users can specify the `-tta` parameter to enable it during evaluation. Take `YOLOv5-s` as an example, its single GPU TTA test command is as follows

```shell
python tools/test.py configs/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739-b804c1ad.pth  --tta
```

For TTA to work properly, you must ensure that the variables `tta_model` and `tta_pipeline` are present in the configuration, see [det_p5_tta.py](https://github.com/open-mmlab/mmyolo/blob/dev/configs/_base_/det_p5_tta.py) for details.

The default TTA in MMYOLO performs 3 multi-scale enhancements, followed by 2 horizontal flip enhancements, for a total of 6 parallel pipelines. take `YOLOv5-s` as an example, its TTA configuration is as follows

```python
img_scales = [(640, 640), (320, 320), (960, 960)]

_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=s),
            dict(
                type='LetterResize',
                scale=s,
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]) for s in img_scales
]

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            _multiscale_resize_transforms,
            [
                dict(type='mmdet.RandomFlip', prob=1.),
                dict(type='mmdet.RandomFlip', prob=0.)
            ], [dict(type='mmdet.LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'pad_param', 'flip',
                               'flip_direction'))
            ]
        ])
]
```

The schematic diagram is shown below.

```text
                               LoadImageFromFile
                    /                 |                     \
(RatioResize,LetterResize) (RatioResize,LetterResize) (RatioResize,LetterResize)
       /      \                    /      \                    /        \
 RandomFlip RandomFlip      RandomFlip RandomFlip        RandomFlip RandomFlip
     |          |                |         |                  |         |
 LoadAnn    LoadAnn           LoadAnn    LoadAnn           LoadAnn    LoadAnn
     |          |                |         |                  |         |
 PackDetIn  PackDetIn         PackDetIn  PackDetIn        PackDetIn  PackDetIn
```

You can modify `img_scales` to support different multi-scale enhancements, or you can insert a new pipeline to implement custom TTA requirements. Assuming you only want to do horizontal flip enhancements, the configuration should be modified as follows.

```python
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='mmdet.RandomFlip', prob=1.),
                dict(type='mmdet.RandomFlip', prob=0.)
            ], [dict(type='mmdet.LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'pad_param', 'flip',
                               'flip_direction'))
            ]
        ])
]
```
