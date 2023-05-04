from mmdet.models.test_time_augs.det_tta import DetTTAModel
from mmcv.transforms.wrappers import Compose
from mmyolo.datasets.transforms.transforms import LetterResize, YOLOv5KeepRatioResize
from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import TestTimeAug
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.formatting import PackDetInputs
# TODO: Need to solve the problem of multiple backend_args parameters
# _backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))

_backend_args = None

tta_model = dict(
    type=DetTTAModel,
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300))

img_scales = [(640, 640), (320, 320), (960, 960)]

#                                LoadImageFromFile
#                     /                 |                     \
# (RatioResize,LetterResize) (RatioResize,LetterResize) (RatioResize,LetterResize) # noqa
#        /      \                    /      \                    /        \
#  RandomFlip RandomFlip      RandomFlip RandomFlip        RandomFlip RandomFlip # noqa
#      |          |                |         |                  |         |
#  LoadAnn    LoadAnn           LoadAnn    LoadAnn           LoadAnn    LoadAnn
#      |          |                |         |                  |         |
#  PackDetIn  PackDetIn         PackDetIn  PackDetIn        PackDetIn  PackDetIn # noqa

_multiscale_resize_transforms = [
    dict(
        type=Compose,
        transforms=[
            dict(type=YOLOv5KeepRatioResize, scale=s),
            dict(
                type=LetterResize,
                scale=s,
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]) for s in img_scales
]

tta_pipeline = [
    dict(type=LoadImageFromFile, backend_args=_backend_args),
    dict(
        type=TestTimeAug,
        transforms=[
            _multiscale_resize_transforms,
            [dict(type=RandomFlip, prob=1.),
             dict(type=RandomFlip, prob=0.)],
            [dict(type=LoadAnnotations, with_bbox=True)],
            [
                dict(
                    type=PackDetInputs,
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'pad_param', 'flip',
                               'flip_direction'))
            ]
        ])
]
