# 测试时增强相关说明

## 测试时增强 TTA

MMYOLO 在 v0.5.0+ 版本中增加对 TTA 的支持，用户可以在进行评估时候指定 `--tta` 参数使用。 以 `YOLOv5-s` 为例，其单卡 TTA 测试命令为：

```shell
python tools/test.py configs/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739-b804c1ad.pth  --tta
```

TTA 功能的正常运行必须确保配置中存在 `tta_model` 和 `tta_pipeline` 两个变量，详情可以参考 [det_p5_tta.py](https://github.com/open-mmlab/mmyolo/blob/dev/configs/_base_/det_p5_tta.py)。

MMYOLO 中默认的 TTA 会先执行 3 个多尺度增强，然后在每个尺度中执行 2 种水平翻转增强，一共 6 个并行的 pipeline。以 `YOLOv5-s` 为例，其 TTA 配置为：

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

其示意图如下所示：

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

你可以修改 `img_scales` 来支持不同的多尺度增强，也可以插入新的 pipeline 从而实现自定义 TTA 需求。 假设你只想进行水平翻转增强，则配置应该修改为如下：

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
