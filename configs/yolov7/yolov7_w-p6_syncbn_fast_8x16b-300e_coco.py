_base_ = './yolov7_l_syncbn_fast_8x16b-300e_coco.py'

img_scale = (1280, 1280)  # height, width
num_classes = 80
# only on Val
batch_shapes_cfg = dict(img_size=img_scale[0], size_divisor=64)

anchors = [
    [(19, 27), (44, 40), (38, 94)],  # P3/8
    [(96, 68), (86, 152), (180, 137)],  # P4/16
    [(140, 301), (303, 264), (238, 542)],  # P5/32
    [(436, 615), (739, 380), (925, 792)]  # P6/64
]
strides = [8, 16, 32, 64]
num_det_layers = 4

model = dict(
    backbone=dict(arch='W', out_indices=(2, 3, 4, 5)),
    neck=dict(
        in_channels=[256, 512, 768, 1024],
        out_channels=[128, 256, 384, 512],
        use_maxpool_in_downsample=False,
        use_repconv_outs=False),
    bbox_head=dict(
        head_module=dict(
            type='YOLOv7p6HeadModule',
            in_channels=[128, 256, 384, 512],
            featmap_strides=strides,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        prior_generator=dict(base_sizes=anchors, strides=strides),
        obj_level_weights=[4.0, 1.0, 0.25, 0.06]))

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader
