_base_ = './ppyoloe_s_fast_8xb32-300e_coco.py'

deepen_factor = 0.67
widen_factor = 0.75

# TODO: training on ppyoloe need to be implemented.
train_batch_size_per_gpu = 28

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
