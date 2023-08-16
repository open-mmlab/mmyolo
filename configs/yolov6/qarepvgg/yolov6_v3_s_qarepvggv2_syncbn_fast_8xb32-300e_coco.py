_base_ = '../yolov6_v3_s_syncbn_fast_8xb32-300e_coco.py'

model = dict(
    backbone=dict(block_cfg=dict(type='QARepVGGBlockV2')),
    neck=dict(block_cfg=dict(type='QARepVGGBlockV2')))
