_base_ = './rtmdet-ins_l_syncbn_fast_8xb32-300e_coco.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'  # noqa

# ========================modified parameters======================
deepen_factor = 0.33
widen_factor = 0.5
img_scale = _base_.img_scale

# ratio range for random resize
random_resize_ratio_range = (0.5, 2.0)
# Number of cached images in mosaic
mosaic_max_cached_images = 40
# Number of cached images in mixup
mixup_max_cached_images = 20

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        # Since the checkpoint includes CUDA:0 data,
        # it must be forced to set map_location.
        # Once checkpoint is fixed, it can be removed.
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=checkpoint,
            map_location='cpu')),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
