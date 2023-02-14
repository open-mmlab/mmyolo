_base_ = './yolov8_m_syncbn_fast_8xb16-500e_coco.py'

# ========================modified parameters======================
deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512

mixup_prob = 0.15

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels]),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels])))

pre_transform = _base_.pre_transform
albu_train_transform = _base_.albu_train_transform
mosaic_affine_pipeline = _base_.mosaic_affine_pipeline
last_transform = _base_.last_transform

train_pipeline = [
    *pre_transform, *mosaic_affine_pipeline,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_pipeline]),
    *last_transform
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
