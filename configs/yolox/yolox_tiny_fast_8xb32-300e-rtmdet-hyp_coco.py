_base_ = './yolox_s_fast_8xb32-300e-rtmdet-hyp_coco.py'

# ========================modified parameters======================
deepen_factor = 0.33
widen_factor = 0.375

# Multi-scale training intervals
# 10 -> 1
batch_augments_interval = 1

scaling_ratio_range = (0.5, 1.5)

# =======================Unmodified in most cases==================
img_scale = _base_.img_scale
pre_transform = _base_.pre_transform

# model settings
model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(
            type='YOLOXBatchSyncRandomResize',
            random_size_range=(320, 640),
            size_divisor=32,
            interval=batch_augments_interval)
    ]),
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_pipeline_stage1 = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=scaling_ratio_range,  # note
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.Resize', scale=(416, 416), keep_ratio=True),  # note
    dict(
        type='mmdet.Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline_stage1))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
