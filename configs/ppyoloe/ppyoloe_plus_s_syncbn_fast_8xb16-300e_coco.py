_base_ = './ppyoloe_s_syncbn_fast_8xb16-300e_coco.py'

model = dict(
    data_preprocessor=dict(mean=[0., 0., 0.], std=[255., 255., 255.]),
    backbone=dict(use_alpha=True))
