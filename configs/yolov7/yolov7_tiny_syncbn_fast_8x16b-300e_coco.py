_base_ = './yolov7_l_syncbn_fast_8x16b-300e_coco.py'

anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32
]

model = dict(
    backbone=dict(arch='Tiny',
                  act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    neck=dict(
        in_channels=[640, 1280, 1280],
        out_channels=[160, 320, 640],
        block_cfg=dict(
            type='ELANBlock',
            mid_ratio=0.4,
            block_ratio=0.4,
            out_ratio=0.5,
            num_blocks=3,
            num_convs_in_block=2),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        use_repconv_outs=False),
    bbox_head=dict(head_module=dict(in_channels=[320, 640, 1280]),
                   prior_generator=dict(base_sizes=anchors)))
