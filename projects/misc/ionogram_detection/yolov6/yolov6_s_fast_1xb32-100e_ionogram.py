_base_ = 'mmyolo::yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py'

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

train_batch_size_per_gpu = 32
train_num_workers = 8

tta_model = None
tta_pipeline = None

# -----train val related-----
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth'  # noqa
#  base_lr_default * (your_bs 32 / default_bs (8 x 32))
base_lr = _base_.base_lr * train_batch_size_per_gpu / (8 * 32)
max_epochs = 100
save_epoch_intervals = 10
val_begin = 20
max_keep_ckpts = 1
log_interval = 50
visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='WandbVisBackend')])

# ==================== Unmodified in most cases ===================
train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=val_begin,
    val_interval=save_epoch_intervals,
    dynamic_intervals=None)

model = dict(
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        initial_assigner=dict(num_classes=num_classes),
        assigner=dict(num_classes=num_classes)))

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
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix)))

test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=test_ann_file,
        data_prefix=dict(img=test_data_prefix)))

val_evaluator = dict(ann_file=data_root + val_data_prefix)
test_evaluator = dict(ann_file=data_root + test_data_prefix)

optim_wrapper = dict(optimizer=dict(lr=base_lr))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=max_keep_ckpts,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=log_interval))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - _base_.num_last_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
]
