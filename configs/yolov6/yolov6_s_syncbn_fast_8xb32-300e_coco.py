_base_ = './yolov6_s_syncbn_fast_8xb32-400e_coco.py'

max_epochs = 300
num_last_epochs = 15

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.01,
        max_epochs=max_epochs))

custom_hooks = [dict(switch_epoch=max_epochs - num_last_epochs)]

train_cfg = dict(
    max_epochs=max_epochs,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
