_base_ = './yolov7_w-p6_syncbn_fast_8x16b-300e_coco.py'

model = dict(
    backbone=dict(arch='E'),
    neck=dict(
        use_maxpool_in_downsample=True,
        use_in_channels_in_downsample=True,
        block_cfg=dict(
            type='ELANBlock',
            middle_ratio=0.4,
            block_ratio=0.2,
            num_blocks=6,
            num_convs_in_block=1),
        in_channels=[320, 640, 960, 1280],
        out_channels=[160, 320, 480, 640]),
    bbox_head=dict(
        head_module=dict(
            in_channels=[160, 320, 480, 640],
            main_out_channels=[320, 640, 960, 1280])))
