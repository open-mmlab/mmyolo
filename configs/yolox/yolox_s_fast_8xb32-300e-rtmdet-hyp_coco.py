_base_ = './yolox_s_fast_8xb8-300e_coco.py'

# ========================modified parameters======================
# Batch size of a single GPU during training
# 8 -> 32
train_batch_size_per_gpu = 32

# Multi-scale training intervals
# 10 -> 1
batch_augments_interval = 1

# Last epoch number to switch training pipeline
# 15 -> 20
num_last_epochs = 20

# Base learning rate for optim_wrapper. Corresponding to 8xb32=256 bs
base_lr = 0.004

# SGD -> AdamW
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# 0.0001 -> 0.0002
ema_momentum = 0.0002

# ============================== Unmodified in most cases ===================
model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(
            type='YOLOXBatchSyncRandomResize',
            random_size_range=(480, 800),
            size_divisor=32,
            interval=batch_augments_interval)
    ]))

param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=_base_.max_epochs - num_last_epochs,
        end=_base_.max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last num_last_epochs epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=_base_.max_epochs - num_last_epochs,
        end=_base_.max_epochs,
    )
]

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        new_train_pipeline=_base_.train_pipeline_stage2,
        priority=48),
    dict(type='mmdet.SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=ema_momentum,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

train_dataloader = dict(batch_size=train_batch_size_per_gpu)
train_cfg = dict(dynamic_intervals=[(_base_.max_epochs - num_last_epochs, 1)])
auto_scale_lr = dict(base_batch_size=8 * train_batch_size_per_gpu)
