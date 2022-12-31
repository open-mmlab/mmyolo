_base_ = './ppyoloe_plus_s_fast_8xb8-80e_coco.py'

deepen_factor = 1.0
widen_factor = 1.0

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

load_from = 'https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_pretrain/ppyoloe_plus_l_obj365_pretrained-3dd89562.pth'  # noqa
