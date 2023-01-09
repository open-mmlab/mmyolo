_base_ = './ppyoloe_plus_s_fast_8xb8-80e_coco.py'

# The pretrained model is geted and converted from official PPYOLOE.
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/README.md
load_from = 'https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_pretrain/ppyoloe_plus_x_obj365_pretrained-43a8000d.pth'  # noqa

deepen_factor = 1.33
widen_factor = 1.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
