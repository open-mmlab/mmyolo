_base_ = '../yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    bbox_head=dict(prior_generator=dict(type='YOLOAutoAnchorGenerator')))

custom_hooks = [
    dict(
        type='YOLOAutoAnchorHook',
        optimizer=dict(
            type='YOLOKMeansAnchorOptimizer',
            iters=1000,
            num_anchor_per_level=[3, 3, 3]))
]
