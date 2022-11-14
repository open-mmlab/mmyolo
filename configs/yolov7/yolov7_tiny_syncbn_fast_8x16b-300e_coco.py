_base_ = './yolov7_l_syncbn_fast_8x16b-300e_coco.py'

anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32
]

model = dict(
    backbone=dict(
        arch='Tiny', act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    neck=dict(
        is_tiny_version_spp=True,
        in_channels=[128, 256, 512],
        out_channels=[64, 128, 256],
        block_cfg=dict(
            _delete_=True,
            type='TinyDownSampleBlock',
            mid_ratio=0.25,
            with_maxpool=False),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        use_repconv_outs=False),
    bbox_head=dict(
        head_module=dict(in_channels=[128, 256, 512]),
        prior_generator=dict(base_sizes=anchors)))
