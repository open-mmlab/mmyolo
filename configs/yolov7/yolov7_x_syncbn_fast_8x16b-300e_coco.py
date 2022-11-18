_base_ = './yolov7_l_syncbn_fast_8x16b-300e_coco.py'

model = dict(
    backbone=dict(arch='X'),
    neck=dict(
        in_channels=[640, 1280, 1280],
        out_channels=[160, 320, 640],
        block_cfg=dict(
            type='ELANBlock',
            middle_ratio=0.4,
            block_ratio=0.4,
            num_blocks=3,
            num_convs_in_block=2),
        use_repconv_outs=False),
    bbox_head=dict(head_module=dict(in_channels=[320, 640, 1280])))
