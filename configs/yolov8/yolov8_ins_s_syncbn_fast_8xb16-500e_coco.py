_base_ = './yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py'

# Batch size of a single GPU during validation
val_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 8

batch_shapes_cfg = dict(
    _delete_=True,
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=_base_.img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5)

model_test_cfg = dict(
    multi_label=True,
    nms_pre=30000,
    min_bbox_size=0,
    score_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.7),
    max_per_img=300,
    mask_thr_binary=0.5)

# ===============================Unmodified in most cases====================
model = dict(
    bbox_head=dict(
        type='YOLOv8InsHead',
        head_module=dict(
            type='YOLOv8InsHeadModule', masks_channels=32,
            protos_channels=256)),
    test_cfg=model_test_cfg)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(batch_shapes_cfg=batch_shapes_cfg))
test_dataloader = val_dataloader

val_evaluator = dict(metric=['bbox', 'segm'])
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
