if '_base_':
    from .._base_.default_runtime import *
    from .._base_.det_p5_tta import *
from mmyolo.models.detectors.yolo_detector import YOLODetector
from mmyolo.models.data_preprocessors.data_preprocessor import PPYOLOEBatchRandomResize, PPYOLOEDetDataPreprocessor
from mmyolo.models.backbones.csp_resnet import PPYOLOECSPResNet
from mmyolo.models.layers.yolo_bricks import EffectiveSELayer, PPYOLOEBasicBlock
from torch.nn.modules.activation import SiLU
from mmyolo.models.necks.ppyoloe_csppan import PPYOLOECSPPAFPN
from mmyolo.models.dense_heads.ppyoloe_head import PPYOLOEHeadModule, PPYOLOEHead
from mmdet.models.task_modules.prior_generators.point_generator import MlvlPointGenerator
from mmyolo.models.task_modules.coders.distance_point_bbox_coder import DistancePointBBoxCoder
from mmdet.models.losses.varifocal_loss import VarifocalLoss
from mmyolo.models.losses.iou_loss import IoULoss
from mmdet.models.losses.gfocal_loss import DistributionFocalLoss
from mmyolo.models.task_modules.assigners.batch_atss_assigner import BatchATSSAssigner
from mmdet.models.task_modules.assigners.iou2d_calculator import BboxOverlaps2D
from mmyolo.models.task_modules.assigners.batch_task_aligned_assigner import BatchTaskAlignedAssigner
from mmcv.transforms.loading import LoadImageFromFile
from mmyolo.datasets.transforms.transforms import PPYOLOERandomDistort, LoadAnnotations, PPYOLOERandomCrop
from mmdet.datasets.transforms import LoadAnnotations as MMDet_LoadAnnotations
from mmdet.datasets.transforms.transforms import RandomFlip, Expand, FixShapeResize
from mmdet.datasets.transforms.formatting import PackDetInputs
from mmengine.dataset.sampler import DefaultSampler
from mmyolo.datasets.utils import yolov5_collate
from mmyolo.datasets.yolov5_coco import YOLOv5CocoDataset
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from torch.optim.sgd import SGD
from mmyolo.engine.hooks.ppyoloe_param_scheduler_hook import PPYOLOEParamSchedulerHook
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.hooks.ema_hook import EMAHook
from mmyolo.models.layers.ema import ExpMomentumEMA
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop

# dataset settings
data_root = '/home/PJLAB/huanghaian/dataset/coco100/'

# parameters that often need to be modified
img_scale = (640, 640)  # width, height
deepen_factor = 0.33
widen_factor = 0.5
max_epochs = 80
num_classes = 80
save_epoch_intervals = 5
train_batch_size_per_gpu = 2
train_num_workers = 8
val_batch_size_per_gpu = 1
val_num_workers = 2

# The pretrained model is geted and converted from official PPYOLOE.
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/README.md
load_from = 'https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_pretrain/ppyoloe_plus_s_obj365_pretrained-bcfe8478.pth'  # noqa

# persistent_workers must be False if num_workers is 0.
persistent_workers = True

# Base learning rate for optim_wrapper
base_lr = 0.001

strides = [8, 16, 32]

model = dict(
    type=YOLODetector,
    data_preprocessor=dict(
        # use this to support multi_scale training
        type=PPYOLOEDetDataPreprocessor,
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type=PPYOLOEBatchRandomResize,
                random_size_range=(320, 800),
                interval=1,
                size_divisor=32,
                random_interp=True,
                keep_ratio=False)
        ],
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type=PPYOLOECSPResNet,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        block_cfg=dict(type=PPYOLOEBasicBlock, shortcut=True, use_alpha=True),
        norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
        act_cfg=dict(type=SiLU, inplace=True),
        attention_cfg=dict(
            type=EffectiveSELayer, act_cfg=dict(type='HSigmoid')),
        use_large_stem=True),
    neck=dict(
        type=PPYOLOECSPPAFPN,
        in_channels=[256, 512, 1024],
        out_channels=[192, 384, 768],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        num_csplayer=1,
        num_blocks_per_layer=3,
        block_cfg=dict(
            type=PPYOLOEBasicBlock, shortcut=False, use_alpha=False),
        norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
        act_cfg=dict(type=SiLU, inplace=True),
        drop_block_cfg=None,
        use_spp=True),
    bbox_head=dict(
        type=PPYOLOEHead,
        head_module=dict(
            type=PPYOLOEHeadModule,
            num_classes=num_classes,
            in_channels=[192, 384, 768],
            widen_factor=widen_factor,
            featmap_strides=strides,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
            act_cfg=dict(type=SiLU, inplace=True),
            num_base_priors=1),
        prior_generator=dict(
            type=MlvlPointGenerator, offset=0.5, strides=strides),
        bbox_coder=dict(type=DistancePointBBoxCoder),
        loss_cls=dict(
            type=VarifocalLoss,
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type=IoULoss,
            iou_mode='giou',
            bbox_format='xyxy',
            reduction='mean',
            loss_weight=2.5,
            return_iou=False),
        # Since the dflloss is implemented differently in the official
        # and mmdet, we're going to divide loss_weight by 4.
        loss_dfl=dict(
            type=DistributionFocalLoss, reduction='mean',
            loss_weight=0.5 / 4)),
    train_cfg=dict(
        initial_epoch=30,
        initial_assigner=dict(
            type=BatchATSSAssigner,
            num_classes=num_classes,
            topk=9,
            iou_calculator=dict(type=BboxOverlaps2D)),
        assigner=dict(
            type=BatchTaskAlignedAssigner,
            num_classes=num_classes,
            topk=13,
            alpha=1,
            beta=6,
            eps=1e-9)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=1000,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=PPYOLOERandomDistort),
    dict(type=Expand, mean=(103.53, 116.28, 123.675)),
    dict(type=PPYOLOERandomCrop),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=yolov5_collate, use_ms_training=True),
    dataset=dict(
        type=YOLOv5CocoDataset,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=train_pipeline))

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(
        type=FixShapeResize,
        width=img_scale[0],
        height=img_scale[1],
        keep_ratio=False,
        interpolation='bicubic'),
    dict(type=MMDet_LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=YOLOv5CocoDataset,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img='val2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        ann_file='annotations/instances_val2017.json',
        pipeline=test_pipeline))

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=SGD, lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=False),
    paramwise_cfg=dict(norm_decay_mult=0.))

default_hooks.merge(
    dict(
        param_scheduler=dict(
            type=PPYOLOEParamSchedulerHook,
            warmup_min_iter=1000,
            start_factor=0.,
            warmup_epochs=5,
            min_lr_ratio=0.0,
            total_epochs=int(max_epochs * 1.2)),
        checkpoint=dict(
            type=CheckpointHook,
            interval=save_epoch_intervals,
            save_best='auto',
            max_keep_ckpts=3)))

custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=ExpMomentumEMA,
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

val_evaluator = dict(
    type=CocoMetric,
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type=EpochBasedTrainLoop,
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)
