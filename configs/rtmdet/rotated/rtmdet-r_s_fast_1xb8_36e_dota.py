_base_ = './rtmdet-r_l_syncbn_fast_1xb8_36e_dota.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'  # noqa

# ========================modified parameters======================
deepen_factor = 0.33
widen_factor = 0.5

# Batch size of a single GPU during training
train_batch_size_per_gpu = 8

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint)),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_dataloader = dict(batch_size=train_batch_size_per_gpu)
