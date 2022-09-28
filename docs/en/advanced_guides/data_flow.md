# Mixed image data augmentation update

Mixed image data augmentation is similar to Mosaic and MixUp, in which the annotation information of multiple images needs to be obtained for fusion during the running process. In the OpenMMLab data augmentation pipeline, other indexes of the dataset are generally not available. In order to achieve the above function, in the YOLOX reproduced in MMDetection, the concept of [MultiImageMixDataset](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/dataset_wrappers.py#L338) dataset wrapper is proposed.

`MultiImageMixDataset` dataset wrapper will include some data augmentation methods such as `Mosaic` and `RandAffine`, while `CocoDataset` will also need to include a `pipeline` to achieve the image and annotation loading function. In this way, we can achieve mixed data augmentation quickly. The configuration method is as follows:

```python
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    ...
]
train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    pipeline=train_pipeline)

```

However, this implementation has a disadvantage: users unfamiliar with MMDetection will forget those data augmentation methods like Mosaic must be used together with `MultiImageMixDataset`, increasing the usage complexity. Moreover, it is hard to understand as well.

To address this problem, further simplifications are made in MMYOLO, which directly lets `pipeline` get `dataset`. In this way, the implementation of `Mosaic` and other data augmentation methods can be achieved and used just as the random flip, without a data wrapper anymore. The new configuration method is as follows:

```python
pre_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True)
]
train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='YOLOXMixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=pre_transform),
    ...
]
```

A more complex YOLOv5-m configuration including MixUp is shown as follows:

```python
mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

# enable mixup
train_pipeline = [
    *pre_transform, *mosaic_affine_pipeline,
    dict(
        type='YOLOv5MixUp',
        prob=0.1,
        pre_transform=[*pre_transform, *mosaic_affine_pipeline]),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
```

It is very easy to use, just pass the object of Dataset to the pipeline.

```python
def prepare_data(self, idx) -> Any:
   """Pass the dataset to the pipeline during training to support mixed
   data augmentation, such as Mosaic and MixUp."""
   if self.test_mode is False:
        data_info = self.get_data_info(idx)
        data_info['dataset'] = self
        return self.pipeline(data_info)
    else:
        return super().prepare_data(idx)
```
