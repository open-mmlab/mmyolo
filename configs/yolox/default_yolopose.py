default_scope = 'mmyolo'

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=35, norm_type=2))

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best=['auto'],
        greater_keys=['coco/ap', 'coco/bbox_mAP']),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
keypoint_colors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 0, 85),
                   (255, 0, 170), (0, 255, 0), (85, 255, 0), (170, 255, 0),
                   (0, 255, 85), (0, 255, 170), (0, 0, 255), (85, 0, 255),
                   (170, 0, 255), (0, 85, 255), (0, 170, 255), (255, 255, 0),
                   (255, 255, 85)]

skeleton_links = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7),
                  (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
                  (11, 13), (13, 15), (12, 14), (14, 16)]

# 18 links
skeleton_links_colors = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 0, 85), (255, 0, 170),
    (0, 255, 0), (85, 255, 0), (170, 255, 0), (0, 255, 85), (0, 255, 170),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (0, 85, 255), (0, 170, 255),
    (255, 255, 0), (255, 255, 85), (255, 0, 255)
]

vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]
visualizer = dict(
    # type='mmdet.DetLocalVisualizer',
    type='mmpose.PoseLocalVisualizer',
    vis_backends=vis_backends,
    kpt_color=keypoint_colors,
    skeleton=skeleton_links,
    link_color=skeleton_links_colors,
    name='visualizer')
seed=0
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

file_client_args = dict(backend='disk')