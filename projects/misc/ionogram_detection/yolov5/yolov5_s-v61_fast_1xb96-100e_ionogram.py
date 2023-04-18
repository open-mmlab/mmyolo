_base_ = 'mmyolo::yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

# ======================= Modified parameters =====================
# -----data related-----
data_root = './Iono4311/'
train_ann_file = 'annotations/train.json'
train_data_prefix = 'train_images/'
val_ann_file = 'annotations/val.json'
val_data_prefix = 'val_images/'
test_ann_file = 'annotations/test.json'
test_data_prefix = 'test_images/'
class_name = ('E', 'Es-l', 'Es-c', 'F1', 'F2', 'Spread-F')
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(250, 165, 30), (120, 69, 125), (53, 125, 34), (0, 11, 123),
             (130, 20, 12), (120, 121, 80)])
# Batch size of a single GPU during training
train_batch_size_per_gpu = 96
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8

# -----model related-----
# Basic size of multi-scale prior box
anchors = [[[8, 6], [24, 4], [19, 9]], [[22, 19], [17, 49], [29, 45]],
           [[44, 66], [96, 76], [126, 59]]]

# -----train val related-----
# base_lr_default * (your_bs / default_bs (8x16)) for SGD
base_lr = _base_.base_lr * train_batch_size_per_gpu / (8 * 16)
max_epochs = 100
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

# default_hooks
save_epoch_intervals = 10
logger_interval = 20
max_keep_ckpts = 1

# train_cfg
val_interval = 2
val_begin = 20

tta_model = None
tta_pipeline = None

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='WandbVisBackend')])

# ===================== Unmodified in most cases ==================
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),
        loss_cls=dict(loss_weight=0.5 *
                      (num_classes / 80 * 3 / _base_.num_det_layers))))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=_base_.dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file=train_ann_file,
            data_prefix=dict(img=train_data_prefix),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=_base_.train_pipeline)))

val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)))

test_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=test_ann_file,
        data_prefix=dict(img=test_data_prefix)))

optim_wrapper = dict(optimizer=dict(lr=base_lr))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_param_scheduler=None,  # for yolov5
        interval=save_epoch_intervals,
        max_keep_ckpts=max_keep_ckpts,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=logger_interval))

val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = dict(ann_file=data_root + test_ann_file)

train_cfg = dict(
    max_epochs=max_epochs, val_begin=val_begin, val_interval=val_interval)
