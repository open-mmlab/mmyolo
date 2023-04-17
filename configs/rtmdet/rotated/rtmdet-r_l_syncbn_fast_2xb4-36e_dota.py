_base_ = '../../_base_/default_runtime.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'  # noqa

# ========================Frequently modified parameters======================
# -----data related-----
data_root = 'data/split_ss_dota/'
# Path of train annotation folder
train_ann_file = 'trainval/annfiles/'
train_data_prefix = 'trainval/images/'  # Prefix of train image path
# Path of val annotation folder
val_ann_file = 'trainval/annfiles/'
val_data_prefix = 'trainval/images/'  # Prefix of val image path
# Path of test images folder
test_data_prefix = 'test/images/'

# Submission dir for result submit
submission_dir = './work_dirs/{{fileBasenameNoExtension}}/submission'

num_classes = 15  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 4
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
# persistent_workers must be False if num_workers is 0.
persistent_workers = True

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 1xb8=8 bs
base_lr = 0.00025  # 0.004 / 16
max_epochs = 36  # Maximum training epochs

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # Decode rbox with angle, For RTMDet-R, Defaults to True.
    # When set to True, use rbox coder such as DistanceAnglePointCoder
    # When set to False, use hbox coder such as DistancePointBBoxCoder
    # different setting lead to different AP.
    decode_with_angle=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.05,  # Threshold to filter out boxes.
    nms=dict(type='nms_rotated', iou_threshold=0.1),  # NMS type and threshold
    max_per_img=2000)  # Max number of detections of each image

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (1024, 1024)  # width, height
# ratio for random rotate
random_rotate_ratio = 0.5
# label ids for rect objs
rotate_rect_obj_labels = [9, 11]
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5DOTADataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 8
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 8

# Config of batch shapes. Only on val. Not use in RTMDet-R
batch_shapes_cfg = None

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 1.0
# The scaling factor that controls the width of the network structure
widen_factor = 1.0
# Strides of multi-scale prior box
strides = [8, 16, 32]
# The angle definition for model
angle_version = 'le90'  # le90, le135, oc are available options

norm_cfg = dict(type='BN')  # Normalization config

# -----train val related-----
lr_start_factor = 1.0e-5
dsl_topk = 13  # Number of bbox selected in each level
loss_cls_weight = 1.0
loss_bbox_weight = 2.0
qfl_beta = 2.0  # beta of QualityFocalLoss
weight_decay = 0.05

# Save model checkpoint and validation intervals
save_checkpoint_intervals = 1
# The maximum checkpoints to keep.
max_keep_ckpts = 3
# single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        type='CSPNeXtPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetRotatedHead',
        head_module=dict(
            type='RTMDetRotatedSepBNHeadModule',
            num_classes=num_classes,
            widen_factor=widen_factor,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=strides),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=qfl_beta,
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='mmrotate.RotatedIoULoss',
            mode='linear',
            loss_weight=loss_bbox_weight),
        angle_version=angle_version,
        # Used for angle encode and decode, similar to bbox coder
        angle_coder=dict(type='mmrotate.PseudoAngleCoder'),
        # If true, it will apply loss_bbox on horizontal box, and angle_loss
        # needs to be specified. In this case the loss_bbox should use
        # horizontal box loss e.g. IoULoss. Arg details can be seen in
        # `docs/zh_cn/tutorials/rotated_detection.md`
        use_hbbox_loss=False,
        loss_angle=None),
    train_cfg=dict(
        assigner=dict(
            type='BatchDynamicSoftLabelAssigner',
            num_classes=num_classes,
            topk=dsl_topk,
            iou_calculator=dict(type='mmrotate.RBboxOverlaps2D'),
            # RBboxOverlaps2D doesn't support batch input, use loop instead.
            batch_iou=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=model_test_cfg,
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(
        type='mmrotate.ConvertBoxType',
        box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmrotate.RandomRotate',
        prob=random_rotate_ratio,
        angle_range=180,
        rotate_type='mmrotate.Rotate',
        rect_obj_labels=rotate_rect_obj_labels),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='RegularizeRotatedBox', angle_version=angle_version),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        box_type='qbox',
        _scope_='mmdet'),
    dict(
        type='mmrotate.ConvertBoxType',
        box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img_path=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img_path=val_data_prefix),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=val_pipeline))

val_evaluator = dict(type='mmrotate.DOTAMetric', metric='mAP')

# Inference on val dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# Inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
# test_dataloader = dict(
#     batch_size=val_batch_size_per_gpu,
#     num_workers=val_num_workers,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path=test_data_prefix),
#         test_mode=True,
#         batch_shapes_cfg=batch_shapes_cfg,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='mmrotate.DOTAMetric',
#     format_only=True,
#     merge_patches=True,
#     outfile_prefix=submission_dir)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=lr_start_factor,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        max_keep_ckpts=max_keep_ckpts,  # only keep latest 3 checkpoints
        save_best='auto'))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_checkpoint_intervals)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

visualizer = dict(type='mmrotate.RotLocalVisualizer')
