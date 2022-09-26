_base_ = './yolov5_l-v61_syncbn_fast_8xb16-300e_coco.py'

model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(type='mmdet.DropBlock', drop_prob=0.05, block_size=3),
                stages=(False, False, True, True)),
        ], ))
