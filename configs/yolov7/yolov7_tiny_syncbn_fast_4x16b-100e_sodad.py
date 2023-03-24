_base_ = './yolov7_tiny_syncbn_fast_8x16b-300e_coco.py'

# ========================Frequently modified parameters======================
# -----data related-----
data_root = './data/SODA-D/'
train_ann_file = 'Annotations/train.json'
train_data_prefix = 'Images'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'Annotations/val.json'
val_data_prefix = 'Images'  # Prefix of val image path
test_ann_file = 'Annotations/test.json'
metainfo = dict(
    classes=('people', 'rider', 'bicycle', 'motor', 'vehicle', 'traffic-sign',
             'traffic-light', 'traffic-camera', 'warning-cone'),
    palette=None)

num_classes = 9  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 4

# -----model related-----
# Basic size of multi-scale prior box
anchors = [
    [(12, 16), (19, 36), (40, 28)],  # P3/8
    [(36, 75), (76, 55), (72, 146)],  # P4/16
    [(142, 110), (192, 243), (459, 401)]  # P5/32
]
# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
# base_lr = base_lr / num_gpus
base_lr = 0.01 / 8
max_epochs = 100  # Maximum training epochs

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (1216, 1216)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32)

# -----train val related-----
loss_cls_weight = 0.3
loss_bbox_weight = 0.05
loss_obj_weight = 0.7

# ===============================Unmodified in most cases====================
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),
        # scaled based on number of detection layers
        loss_cls=dict(loss_weight=loss_cls_weight *
                      (num_classes / 80 * 3 / _base_.num_det_layers)),
        loss_bbox=dict(
            loss_weight=loss_bbox_weight * (3 / _base_.num_det_layers),
            return_iou=True),
        loss_obj=dict(loss_weight=loss_obj_weight *
                      ((img_scale[0] / 640)**2 * 3 / _base_.num_det_layers)),
    ))



pre_transform = _base_.pre_transform
max_translate_ratio = _base_.max_translate_ratio
scaling_ratio_range = _base_.scaling_ratio_range
randchoice_mosaic_prob = _base_.randchoice_mosaic_prob
mixup_alpha = _base_.mixup_alpha
mixup_beta = _base_.mixup_beta
mixup_prob = _base_.mixup_prob

mosiac4_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,  # change
        scaling_ratio_range=scaling_ratio_range,  # change
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
]
mosiac9_pipeline = [
    dict(
        type='Mosaic9',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,  # change
        scaling_ratio_range=scaling_ratio_range,  # change
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
]

randchoice_mosaic_pipeline = dict(
    type='RandomChoice',
    transforms=[mosiac4_pipeline, mosiac9_pipeline],
    prob=randchoice_mosaic_prob)

train_pipeline = [
    *pre_transform,
    randchoice_mosaic_pipeline,
    dict(
        type='YOLOv5MixUp',
        alpha=mixup_alpha,
        beta=mixup_beta,
        prob=mixup_prob,  # change
        pre_transform=[*pre_transform, randchoice_mosaic_pipeline]),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix)))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(img=val_data_prefix),
        ann_file=test_ann_file,
        batch_shapes_cfg=batch_shapes_cfg))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.9,
        weight_decay=_base_.weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    clip_grad=None,
    constructor='YOLOv7OptimWrapperConstructor')

default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs))
val_evaluator = dict(type='SodadMetric', classwise=True, ann_file=data_root + val_ann_file)
test_evaluator = dict(type='SodadMetric', classwise=True, ann_file=data_root + test_ann_file)

train_cfg = dict(
    max_epochs=max_epochs,
    dynamic_intervals=[(max_epochs - _base_.num_epoch_stage2,
                        _base_.val_interval_stage2)])
