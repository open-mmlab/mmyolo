_base_ = './yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py'

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

val_evaluator = dict(metric=['bbox', 'segm'])
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
