_base_ = './ppyoloe_s_syncbn_fast_8xb32-300e_coco.py'

deepen_factor = 0.33
widen_factor = 0.50
train_batch_size_per_gpu = 28
static_assigner_epoch = 133

model = dict(
    backbone=dict(
        type='CSPResNet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor), ))

test_pipeline = _base_.test_pipeline

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

test_dataloader = val_dataloader
