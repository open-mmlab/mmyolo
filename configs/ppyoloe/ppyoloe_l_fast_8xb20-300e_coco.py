_base_ = './ppyoloe_s_fast_8xb32-300e_coco.py'

deepen_factor = 1.0
widen_factor = 1.0

train_batch_size_per_gpu = 20

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_dataloader = dict(batch_size=train_batch_size_per_gpu)

load_from = 'https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_pretrain/cspresnet_l_imagenet1k_pretrained-c0010e6c.pth'  # noqa
