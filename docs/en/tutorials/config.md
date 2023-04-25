# Learn about Configs with YOLOv5

MMYOLO and other OpenMMLab repositories use [MMEngine's config system](https://mmengine.readthedocs.io/en/latest/tutorials/config.html). It has a modular and inheritance design, which is convenient to conduct various experiments.

## Config file content

MMYOLO uses a modular design, all modules with different functions can be configured through the config. Taking [yolov5_s-v61_syncbn_8xb16-300e_coco.py](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py) as an example, we will introduce each field in the config according to different function modules:

### Important parameters

When changing the training configuration, it is usually necessary to modify the following parameters. For example, the scaling factors `deepen_factor` and `widen_factor` are used by the network to control the size of the model in MMYOLO. So we recommend defining these parameters separately in the configuration file.

```python
img_scale = (640, 640)            # height of image, width of image
deepen_factor = 0.33              # The scaling factor that controls the depth of the network structure, 0.33 for YOLOv5-s
widen_factor = 0.5                # The scaling factor that controls the width of the network structure, 0.5 for YOLOv5-s
max_epochs = 300                  # Maximum training epochs: 300 epochs
save_epoch_intervals = 10         # Validation intervals. Run validation every 10 epochs.
train_batch_size_pre_gpu = 16     # Batch size of a single GPU during training
train_num_workers = 8             # Worker to pre-fetch data for each single GPU
val_batch_size_pre_gpu = 1        # Batch size of a single GPU during validation.
val_num_workers = 2               # Worker to pre-fetch data for each single GPU during validation
```

### Model config

In MMYOLO's config, we use `model` to set up detection algorithm components. In addition to neural network components such as `backbone`, `neck`, etc, it also requires `data_preprocessor`, `train_cfg`, and `test_cfg`. `data_preprocessor` is responsible for processing a batch of data output by the dataloader. `train_cfg` and `test_cfg` in the model config are for training and testing hyperparameters of the components.

```python
anchors = [[(10, 13), (16, 30), (33, 23)], # Basic size of multi-scale prior box
           [(30, 61), (62, 45), (59, 119)],
           [(116, 90), (156, 198), (373, 326)]]
strides = [8, 16, 32] # Strides of multi-scale prior box

model = dict(
    type='YOLODetector', # The name of detector
    data_preprocessor=dict(  # The config of data preprocessor, usually includes image normalization and padding
        type='mmdet.DetDataPreprocessor',  # The type of the data preprocessor, refer to https://mmdetection.readthedocs.io/en/dev-3.x/api.html#module-mmdet.models.data_preprocessors. It is worth noticing that using `YOLOv5DetDataPreprocessor` achieves faster training speed.
        mean=[0., 0., 0.],  # Mean values used to pre-training the pre-trained backbone models, ordered in R, G, B
        std=[255., 255., 255.], # Standard variance used to pre-training the pre-trained backbone models, ordered in R, G, B
        bgr_to_rgb=True),  # whether to convert image from BGR to RGB
    backbone=dict(  # The config of backbone
        type='YOLOv5CSPDarknet',  # The type of backbone, currently it is available candidates are 'YOLOv5CSPDarknet', 'YOLOv6EfficientRep', 'YOLOXCSPDarknet'
        deepen_factor=deepen_factor, # The scaling factor that controls the depth of the network structure
        widen_factor=widen_factor, # The scaling factor that controls the width of the network structure
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), # The config of normalization layers.
        act_cfg=dict(type='SiLU', inplace=True)), # The config of activation function
    neck=dict(
        type='YOLOv5PAFPN',  # The neck of detector is YOLOv5FPN, We also support 'YOLOv6RepPAFPN', 'YOLOXPAFPN'.
        deepen_factor=deepen_factor, # The scaling factor that controls the depth of the network structure
        widen_factor=widen_factor, # The scaling factor that controls the width of the network structure
        in_channels=[256, 512, 1024], # The input channels, this is consistent with the output channels of backbone
        out_channels=[256, 512, 1024], # The output channels of each level of the pyramid feature map, this is consistent with the input channels of head
        num_csp_blocks=3, # The number of bottlenecks of CSPLayer
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), # The config of normalization layers.
        act_cfg=dict(type='SiLU', inplace=True)), # The config of activation function
    bbox_head=dict(
        type='YOLOv5Head', # The type of BBox head is 'YOLOv5Head', we also support 'YOLOv6Head', 'YOLOXHead'
        head_module=dict(
            type='YOLOv5HeadModule', # The type of Head module is 'YOLOv5HeadModule', we also support 'YOLOv6HeadModule', 'YOLOXHeadModule'
            num_classes=80, # Number of classes for classification
            in_channels=[256, 512, 1024], # The input channels, this is consistent with the input channels of neck
            widen_factor=widen_factor, # The scaling factor that controls the width of the network structure
            featmap_strides=[8, 16, 32], # The strides of the multi-scale feature maps
            num_base_priors=3), # The number of prior boxes on a certain point
        prior_generator=dict( # The config of prior generator
            type='mmdet.YOLOAnchorGenerator', # The prior generator uses 'YOLOAnchorGenerator. Refer to https://github.com/open-mmlab/mmdetection/blob/dev-3.x/mmdet/models/task_modules/prior_generators/anchor_generator.py for more details
            base_sizes=anchors, # Basic scale of the anchor
            strides=strides), # The strides of the anchor generator. This is consistent with the FPN feature strides. The strides will be taken as base_sizes if base_sizes is not set.
    ),
    test_cfg=dict(
        multi_label=True, # The config of multi-label for multi-clas prediction. The default setting is True.
        nms_pre=30000,  # The number of boxes before NMS
        score_thr=0.001, # Threshold to filter out boxes.
        nms=dict(type='nms', # Type of NMS
                 iou_threshold=0.65), # NMS threshold
        max_per_img=300)) # Max number of detections of each image
```

### Dataset and evaluator config

[Dataloaders](https://pytorch.org/docs/stable/data.html?highlight=data%20loader#torch.utils.data.DataLoader) are required for the training, validation, and testing of the [runner](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html). Dataset and data pipeline need to be set to build the dataloader. Due to the complexity of this part, we use intermediate variables to simplify the writing of dataloader configs. More complex data augmentation methods are adopted for the lightweight object detection algorithms in MMYOLO. Therefore, MMYOLO has a wider range of dataset configurations than other models in MMDetection.

The training and testing data flow of YOLOv5 have a certain difference. We will introduce them separately here.

```python
dataset_type = 'CocoDataset'  # Dataset type, this will be used to define the dataset
data_root = 'data/coco/'  # Root path of data

pre_transform = [ # Training data loading pipeline
    dict(
        type='LoadImageFromFile'), # First pipeline to load images from file path
    dict(type='LoadAnnotations', # Second pipeline to load annotations for current image
         with_bbox=True) # Whether to use bounding box, True for detection
]

albu_train_transforms = [		     # Albumentation is introduced for image data augmentation. We follow the code of YOLOv5-v6.1, please make sure its version is 1.0.+
    dict(type='Blur', p=0.01),       # Blur augmentation, the probability is 0.01
    dict(type='MedianBlur', p=0.01), # Median blue augmentation, the probability is 0.01
    dict(type='ToGray', p=0.01),	 # Randomly convert RGB to gray-scale image, the probability is 0.01
    dict(type='CLAHE', p=0.01)		 # CLAHE(Limited Contrast Adaptive Histogram Equalization) augmentation, the probability is 0.01
]
train_pipeline = [				# Training data processing pipeline
    *pre_transform,				# Introduce the pre-defined training data loading processing
    dict(
        type='Mosaic',          # Mosaic augmentation
        img_scale=img_scale,    # The image scale after Mosaic augmentation
        pad_val=114.0,          # Pixel values filled with empty areas
        pre_transform=pre_transform), # Pre-defined training data loading pipeline
    dict(
        type='YOLOv5RandomAffine',	    # Random Affine augmentation for YOLOv5
        max_rotate_degree=0.0,          # Maximum degrees of rotation transform
        max_shear_degree=0.0,           # Maximum degrees of shear transform
        scaling_ratio_range=(0.5, 1.5), # Minimum and maximum ratio of scaling transform
        border=(-img_scale[0] // 2, -img_scale[1] // 2), # Distance from height and width sides of input image to adjust output shape. Only used in mosaic dataset.
        border_val=(114, 114, 114)), # Border padding values of 3 channels.
    dict(
        type='mmdet.Albu',			# Albumentation of MMDetection
        transforms=albu_train_transforms, # Pre-defined albu_train_transforms
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),            # Random augmentation on HSV channel
    dict(type='mmdet.RandomFlip', prob=0.5),	# Random flip, the probability is 0.5
    dict(
        type='mmdet.PackDetInputs',				# Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_dataloader = dict( # Train dataloader config
    batch_size=train_batch_size_pre_gpu, # Batch size of a single GPU during training
    num_workers=train_num_workers, # Worker to pre-fetch data for each single GPU during training
    persistent_workers=True, # If ``True``, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed.
    pin_memory=True, # If ``True``, the dataloader will allow pinned memory, which can reduce copy time between CPU and memory
    sampler=dict( # training data sampler
        type='DefaultSampler', # DefaultSampler which supports both distributed and non-distributed training. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py
        shuffle=True), # randomly shuffle the training data in each epoch
    dataset=dict( # Train dataset config
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json', # Path of annotation file
        data_prefix=dict(img='train2017/'), # Prefix of image path
        filter_cfg=dict(filter_empty_gt=False, min_size=32), # Config of filtering images and annotations
        pipeline=train_pipeline))
```

In the testing phase of YOLOv5, the [Letter Resize](https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/datasets/transforms/transforms.py#L116) method resizes all the test images to the same scale, which preserves the aspect ratio of all testing images. Therefore, the validation and testing phases share the same data pipeline.

```python
test_pipeline = [ # Validation/ Testing dataloader config
    dict(
        type='LoadImageFromFile'), # First pipeline to load images from file path
    dict(type='YOLOv5KeepRatioResize', # Second pipeline to resize images with the same aspect ratio
         scale=img_scale), # Pipeline that resizes the images
    dict(
        type='LetterResize', # Third pipeline to rescale images to meet the requirements of different strides
        scale=img_scale, # Target scale of image
        allow_scale_up=False, # Allow scale up when radio > 1
        pad_val=dict(img=114)), # Padding value
    dict(type='LoadAnnotations', with_bbox=True), # Forth pipeline to load annotations for current image
    dict(
        type='mmdet.PackDetInputs', # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_pre_gpu, # Batch size of a single GPU
    num_workers=val_num_workers, # Worker to pre-fetch data for each single GPU
    persistent_workers=True, # If ``True``, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed.
    pin_memory=True, # If ``True``, the dataloader will allow pinned memory, which can reduce copy time between CPU and memory
    drop_last=False, # IF ``True``, the dataloader will drop data, which fails to make a batch
    sampler=dict(
        type='DefaultSampler', # Default sampler for both distributed and normal training
        shuffle=False), # not shuffle during validation and testing
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True, # # Turn on test mode of the dataset to avoid filtering annotations or images
        data_prefix=dict(img='val2017/'), # Prefix of image path
        ann_file='annotations/instances_val2017.json', # Path of annotation file
        pipeline=test_pipeline,
        batch_shapes_cfg=dict(  # Config of batch shapes
            type='BatchShapePolicy', # Policy that makes paddings with least pixels during batch inference process, which does not require the image scales of all batches to be the same throughout validation.
            batch_size=val_batch_size_pre_gpu, # Batch size for batch shapes strategy, equals to validation batch size on single GPU
            img_size=img_scale[0], # Image scale
            size_divisor=32, # The image scale of padding should be divided by pad_size_divisor
            extra_pad_ratio=0.5))) # additional paddings for pixel scale

test_dataloader = val_dataloader
```

[Evaluators](https://mmengine.readthedocs.io/en/latest/design/evaluation.html) are used to compute the metrics of the trained model on the validation and testing datasets. The config of evaluators consists of one or a list of metric configs:

```python
val_evaluator = dict(  # Validation evaluator config
    type='mmdet.CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection
    proposal_nums=(100, 1, 10),	# The number of proposal used to evaluate for detection
    ann_file=data_root + 'annotations/instances_val2017.json',  # Annotation file path
    metric='bbox',  # Metrics to be evaluated, `bbox` for detection
)
test_evaluator = val_evaluator  # Testing evaluator config
```

Since the test dataset has no annotation files, the test_dataloader and test_evaluator config in MMYOLO are generally the same as the val's. If you want to save the detection results on the test dataset, you can write the config like this:

```python
# inference on test dataset and
# format the output results for submission.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/image_info_test-dev2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/image_info_test-dev2017.json',
    metric='bbox',
    format_only=True,  # Only format and save the results to coco json file
    outfile_prefix='./work_dirs/coco_detection/test')  # The prefix of output json files
```

### Training and testing config

MMEngine's runner uses Loop to control the training, validation, and testing processes.
Users can set the maximum training epochs and validation intervals with these fields.

```python
max_epochs = 300 # Maximum training epochs: 300 epochs
save_epoch_intervals = 10 # Validation intervals. Run validation every 10 epochs.

train_cfg = dict(
    type='EpochBasedTrainLoop',  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=max_epochs,  # Maximum training epochs: 300 epochs
    val_interval=save_epoch_intervals)  # Validation intervals. Run validation every 10 epochs.
val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type
```

MMEngine also supports dynamic intervals for evaluation. For example, you can run validation every 10 epochs on the first 280 epochs, and run validation every epoch on the final 20 epochs. The configurations are as follows.

```python
max_epochs = 300 # Maximum training epochs: 300 epochs
save_epoch_intervals = 10 # Validation intervals. Run validation every 10 epochs.

train_cfg = dict(
    type='EpochBasedTrainLoop',  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=max_epochs,  # Maximum training epochs: 300 epochs
    val_interval=save_epoch_intervals,  # Validation intervals. Run validation every 10 epochs.
    dynamic_intervals=[(280, 1)]) # Switch evaluation on 280 epoch and switch the interval to 1.
val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type
```

### Optimization config

`optim_wrapper` is the field to configure optimization-related settings. The optimizer wrapper not only provides the functions of the optimizer but also supports functions such as gradient clipping, mixed precision training, etc. Find out more in the [optimizer wrapper tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html).

```python
optim_wrapper = dict(  # Optimizer wrapper config
    type='OptimWrapper',  # Optimizer wrapper type, switch to AmpOptimWrapper to enable mixed precision training.
    optimizer=dict(  # Optimizer config. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # Stochastic gradient descent optimizer
        lr=0.01,  # The base learning rate
        momentum=0.937, # Stochastic gradient descent with momentum
        weight_decay=0.0005, # Weight decay of SGD
        nesterov=True, # Enable Nesterov momentum, Refer to http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
        batch_size_pre_gpu=train_batch_size_pre_gpu),  # Enable automatic learning rate scaling
    clip_grad=None,  # Gradient clip option. Set None to disable gradient clip. Find usage in https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html
    constructor='YOLOv5OptimizerConstructor') # The constructor for YOLOv5 optimizer
```

`param_scheduler` is the field that configures methods of adjusting optimization hyperparameters such as learning rate and momentum. Users can combine multiple schedulers to create a desired parameter adjustment strategy. Find more in the [parameter scheduler tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html). In YOLOv5, parameter scheduling is complex to implement and difficult to implement with `param_scheduler`. So we use `YOLOv5ParamSchedulerHook` to implement it (see next section), which is simpler but less versatile.

```python
param_scheduler = None
```

### Hook config

Users can attach hooks to training, validation, and testing loops to insert some operations during running. There are two different hook fields, one is `default_hooks` and the other is `custom_hooks`.

`default_hooks` is a dict of hook configs for the hooks that must be required at the runtime. They have default priority which should not be modified. If not set, the runner will use the default values. To disable a default hook, users can set its config to `None`.

```python
default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook', # MMYOLO uses `YOLOv5ParamSchedulerHook` to adjust hyper-parameters in optimizers
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook', # Hook to save model checkpoint on specific intervals
        interval=save_epoch_intervals, # Save model checkpoint every 10 epochs.
        max_keep_ckpts=3)) # The maximum checkpoints to keep.
```

`custom_hooks` is a list of hook configs. Users can develop their hooks and insert them in this field.

```python
custom_hooks = [
    dict(
        type='EMAHook', # A Hook to apply Exponential Moving Average (EMA) on the model during training.
        ema_type='ExpMomentumEMA', # The type of EMA strategy to use.
        momentum=0.0001, # The momentum of EMA
        update_buffers=True, # # If ``True``, calculate the running averages of model parameters
        priority=49) # Priority higher than NORMAL(50)
]
```

### Runtime config

```python
default_scope = 'mmyolo'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/tutorials/registry.html

env_cfg = dict(
    cudnn_benchmark=True,  # Whether to enable cudnn benchmark
    mp_cfg=dict(  # Multi-processing config
        mp_start_method='fork',  # Use fork to start multi-processing threads. 'fork' is usually faster than 'spawn' but may be unsafe. See discussion in https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0),  # Disable opencv multi-threads to avoid system being overloaded
    dist_cfg=dict(backend='nccl'),  # Distribution configs
)

vis_backends = [dict(type='LocalVisBackend')]  # Visualization backends. Refer to: https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html
visualizer = dict(
    type='mmdet.DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(
    type='LogProcessor',  # Log processor to process runtime logs
    window_size=50,  # Smooth interval of log values
    by_epoch=True)  # Whether to format logs with epoch style. Should be consistent with the train loop's type.

log_level = 'INFO'  # The level of logging.
load_from = None  # Load model checkpoint as a pre-trained model from a given path. This will not resume training.
resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.
```

## Config file inheritance

`config/_base_` contains default runtime. The configs that are composed of components from `_base_` are called _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should be inherited  from the _primitive_ config. In this way, the maximum inheritance level is 3.

For easy understanding, we recommend contributors inherit from existing methods.
For example, if some modification is made based on YOLOv5-s, such as modifying the depth of the network, users may first inherit the `_base_ = ./yolov5_s-v61_syncbn_8xb16-300e_coco.py `, then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods, you may create a folder `yolov100` under `configs`,

Please refer to the [mmengine config tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/config.html) for more details.

By setting the `_base_` field, we can set which files the current configuration file inherits from.

When `_base_` is a string of a file path, it means inheriting the contents of one config file.

```python
_base_ = '../_base_/default_runtime.py'
```

When `_base_` is a list of multiple file paths, it means inheriting multiple files.

```python
_base_ = [
    './yolov5_s-v61_syncbn_8xb16-300e_coco.py',
    '../_base_/default_runtime.py'
]
```

If you wish to inspect the config file, you may run `mim run mmdet print_config /PATH/TO/CONFIG` to see the complete config.

### Ignore some fields in the base configs

Sometimes, you may set `_delete_=True` to ignore some of the fields in base configs.
You may refer to the [mmengine config tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/config.html) for a simple illustration.

In MMYOLO, for example, to change the backbone of RTMDet with the following config.

```python
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(...),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(...),
    bbox_head=dict(...))
```

If you want to change `CSPNeXt` to `YOLOv6EfficientRep` for the RTMDet backbone, because there are different fields (`channel_attention` and `expand_ratio`) in `CSPNeXt` and `YOLOv6EfficientRep`, you need to use `_delete_=True` to replace all the old keys in the `backbone` field with the new keys.

```python
_base_ = '../rtmdet/rtmdet_l_syncbn_8xb32-300e_coco.py'
model = dict(
    backbone=dict(
        _delete_=True,
        type='YOLOv6EfficientRep',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True)),
    neck=dict(...),
    bbox_head=dict(...))
```

### Use intermediate variables in configs

Some intermediate variables are used in the configs files, like `train_pipeline` and `test_pipeline` in datasets. It's worth noting that when modifying intermediate variables in the children configs, users need to pass the intermediate variables into corresponding fields again.
For example, we would like to change the `image_scale` during training and add `YOLOv5MixUp` data augmentation, `img_scale/train_pipeline/test_pipeline` are intermediate variables we would like to modify.

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

img_scale = (1280, 1280)  # image height, image width
affine_scale = 0.9

mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *pre_transform, *mosaic_affine_pipeline,
    dict(
        type='YOLOv5MixUp',	# MixUp augmentation of YOLOv5
        prob=0.1, # the probability of YOLOv5MixUp
        pre_transform=[*pre_transform,*mosaic_affine_pipeline]), # Pre-defined Training data pipeline and MixUp augmentation.
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

test_pipeline = [
    dict(
        type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
```

We first define a new `train_pipeline`/`test_pipeline` and pass it into `data`.

Likewise, if we want to switch from `SyncBN` to `BN` or `MMSyncBN`, we need to modify every `norm_cfg` in the configuration file.

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    ...)
```

### Reuse variables in \_base\_ file

If the users want to reuse the variables in the base file, they can get a copy of the corresponding variable by using `{{_base_.xxx}}`. The latest version of MMEngine also supports reusing variables without `{{}}` usage.

E.g:

```python
_base_ = '../_base_/default_runtime.py'

pre_transform = _base_.pre_transform # `pre_transform` equals to `pre_transform` in the _base_ config
```

## Modify config through script arguments

When submitting jobs using `tools/train.py` or `tools/test.py`, you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `train_dataloader.dataset.pipeline` is normally a list, e.g. `[dict(type='LoadImageFromFile'), ...]`. If you want to change `'LoadImageFromFile'` to `'LoadImageFromNDArray'` in the pipeline, you may specify `--cfg-options data.train.pipeline.0.type=LoadImageFromNDArray`.

- Update values of list/tuples.

  Sometimes the value to update is a list or a tuple, for example, the config file normally sets `model.data_preprocessor.mean=[123.675, 116.28, 103.53]`. If you want to change the mean values, you may specify `--cfg-options model.data_preprocessor.mean="[127,127,127]"`. Note that the quotation mark `"` is necessary to support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Config name style

We follow the below style to name config files. Contributors are advised to follow the same style.

```
{algorithm name}_{model component names [component1]_[component2]_[...]}-[version id]_[norm setting]_[data preprocessor type]_{training settings}_{training dataset information}_[testing dataset information].py
```

The file name is divided into 8 name fields, which have 4 required parts and 4 optional parts. All parts and components are connected with `_` and words of each part or component should be connected with `-`. `{}` indicates the required name field, and `[]` indicates the optional name field.

- `{algorithm name}`: The name of the algorithm. It can be a detector name such as `yolov5`, `yolov6`, `yolox`, etc.
- `{component names}`:  Names of the components used in the algorithm such as backbone, neck, etc. For example, `yolov5_s` means its `deepen_factor` is `0.33` and its `widen_factor` is `0.5`.
- `[version_id]` (optional): Since the evolution of the YOLO series is much faster than traditional object detection algorithms, `version id` is used to distinguish the differences between different sub-versions. E.g, YOLOv5-3.0 uses the `Focus` layer as the stem layer, and YOLOv5-6.0 uses the `Conv` layer as the stem layer.
- `[norm_setting]` (optional): `bn` indicates `Batch Normalization`, `syncbn` indicates `Synchronized Batch Normalization`。
- `[data preprocessor type]` (optional): `fast` incorporates  [YOLOv5DetDataPreprocessor](https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/models/data_preprocessors/data_preprocessor.py#L9)  and [yolov5_collate](https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/datasets/utils.py#L12) to preprocess data. The training speed is faster than the default `mmdet.DetDataPreprocessor`, while results in extending the overall pipeline to multi-task learning.
- `{training settings}`: Information of training settings such as batch size, augmentations, loss trick, scheduler, and epochs/iterations. For example: `8xb16-300e_coco` means using 8-GPUs x 16-images-per-GPU, and train 300 epochs.
  Some abbreviations:
  - `{gpu x batch_per_gpu}`:  GPUs and samples per GPU.  For example, `4xb4` is the short term of 4-GPUs x 4-images-per-GPU.
  - `{schedule}`: training schedule, default option in MMYOLO is 300 epochs.
- `{training dataset information}`: Training dataset names like `coco`, `cityscapes`, `voc-0712`, `wider-face`, and `balloon`.
- `[testing dataset information]` (optional): Testing dataset name for models trained on one dataset but tested on another. If not mentioned, it means the model was trained and tested on the same dataset type.
