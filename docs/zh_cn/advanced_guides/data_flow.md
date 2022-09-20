# 混合类图片数据增强更新

混合类图片数据增强是指类似 Mosaic 和 MixUp 一样，在运行过程中需要获取多张图片的标注信息进行融合。 在 OpenMMLab 数据增强 pipeline 中一般是获取不到数据集其他索引的。 为了实现上述功能，在 MMDetection 复现的 YOLOX 中提出了 [MultiImageMixDataset](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/dataset_wrappers.py#L338) 数据集包装器的概念。

`MultiImageMixDataset` 数据集包装器会传入一个包括 `Mosaic` 和 `RandAffine` 等数据增强，而 `CocoDataset` 中也需要传入一个包括图片和标注加载的 `pipeline` 。通过这种方式就可以快速的实现混合类数据增强。其配置用法如下所示：

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

但是上述实现起来会有一个缺点：对于不熟悉 MMDetection 的用户来说，其经常会忘记 Mosaic 必须要和 `MultiImageMixDataset` 配合使用，而且这样会加大复杂度和理解难度。

为了解决这个问题，在 MMYOLO 中进一步进行了简化。直接让 `pipeline` 获取到 `dataset` 对象，此时就可以将 `Mosaic` 等混合类数据增强的实现和使用随机翻转的操作一样，不再需要数据集包装器。新的配置写法为：

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

一个稍微复杂点的包括 MixUp 的 YOLOv5-m 配置如下所示：

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

其实现过程非常简单，只需要在 Dataset 中将本身对象传给 pipeline 即可，具体代码如下：

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
