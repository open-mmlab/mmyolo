# Replace the backbone network

```{note}
1. When using other backbone networks, you need to ensure that the output channels of the backbone network match the input channels of the neck network.
2. The configuration files given below only ensure that the training will work correctly, and their training performance may not be optimal. Because some backbones require specific learning rates, optimizers, and other hyperparameters. Related contents will be added in the "Training Tips" section later.
```

## Use backbone network implemented in MMYOLO

Suppose you want to use `YOLOv6EfficientRep` as the backbone network of `YOLOv5`, the example config is as the following:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    backbone=dict(
        type='YOLOv6EfficientRep',
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True))
)
```

## Use backbone network implemented in other OpenMMLab repositories

The model registry in MMYOLO, MMDetection, MMClassification, and MMSegmentation all inherit from the root registry in MMEngine in the OpenMMLab 2.0 system, allowing these repositories to directly use modules already implemented by each other. Therefore, in MMYOLO, users can use backbone networks from MMDetection and MMClassification without reimplementation.

### Use backbone network implemented in MMDetection

1. Suppose you want to use `ResNet-50` as the backbone network of `YOLOv5`, the example config is as the following:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [512, 1024, 2048]

model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmdet.ResNet', # Using ResNet from mmdet
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='YOLOv5PAFPN',
        widen_factor=widen_factor,
        in_channels=channels, # Note: The 3 channels of ResNet-50 output are [512, 1024, 2048], which do not match the original yolov5-s neck and need to be changed.
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels, # input channels of head need to be changed accordingly
            widen_factor=widen_factor))
)
```

2. Suppose you want to use `SwinTransformer-Tiny` as the backbone network of `YOLOv5`, the example config is as the following:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [192, 384, 768]
checkpoint_file = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmdet.SwinTransformer', # Using SwinTransformer from mmdet
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=channels, # Note: The 3 channels of SwinTransformer-Tiny output are [192, 384, 768], which do not match the original yolov5-s neck and need to be changed.
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels, # input channels of head need to be changed accordingly
            widen_factor=widen_factor))
)
```

### Use backbone network implemented in MMClassification

1. Suppose you want to use `ConvNeXt-Tiny` as the backbone network of `YOLOv5`, the example config is as the following:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# please run the command, mim install "mmcls>=1.0.0rc2", to install mmcls
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa
deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [192, 384, 768]

model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmcls.ConvNeXt', # Using ConvNeXt from mmcls
        arch='tiny',
        out_indices=(1, 2, 3),
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')), # The pre-trained weights of backbone network in MMCls have prefix='backbone.'. The prefix in the keys will be removed so that these weights can be normally loaded.
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=channels, # Note: The 3 channels of ConvNeXt-Tiny output are [192, 384, 768], which do not match the original yolov5-s neck and need to be changed.
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels, # input channels of head need to be changed accordingly
            widen_factor=widen_factor))
)
```

2. Suppose you want to use `MobileNetV3-small` as the backbone network of `YOLOv5`, the example config is as the following:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# please run the command, mim install "mmcls>=1.0.0rc2", to install mmcls
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_small-8427ecf0.pth'  # noqa
deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [24, 48, 96]

model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmcls.MobileNetV3', # Using MobileNetV3 from mmcls
        arch='small',
        out_indices=(3, 8, 11), # Modify out_indices
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint_file,
            prefix='backbone.')), # The pre-trained weights of backbone network in MMCls have prefix='backbone.'. The prefix in the keys will be removed so that these weights can be normally loaded.
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=channels, # Note: The 3 channels of MobileNetV3 output are [24, 48, 96], which do not match the original yolov5-s neck and need to be changed.
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels, # input channels of head need to be changed accordingly
            widen_factor=widen_factor))
)
```

### Use backbone network in `timm` through MMClassification

MMClassification also provides a wrapper for the Py**T**orch **Im**age **M**odels (`timm`) backbone network, users can directly use the backbone network in `timm` through MMClassification. Suppose you want to use `EfficientNet-B1` as the backbone network of `YOLOv5`, the example config is as the following:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# please run the command, mim install "mmcls>=1.0.0rc2", to install mmcls
# and the command, pip install timm, to install timm
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)

deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [40, 112, 320]

model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmcls.TIMMBackbone', # Using timm from mmcls
        model_name='efficientnet_b1', # Using efficientnet_b1 in timm
        features_only=True,
        pretrained=True,
        out_indices=(2, 3, 4)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=channels, # Note: The 3 channels of EfficientNet-B1 output are [40, 112, 320], which do not match the original yolov5-s neck and need to be changed.
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels, # input channels of head need to be changed accordingly
            widen_factor=widen_factor))
)
```

### Use backbone network implemented in MMSelfSup

Suppose you want to use `ResNet-50` which is self-supervised trained by `MoCo v3` in MMSelfSup as the backbone network of `YOLOv5`, the example config is as the following:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# please run the command, mim install "mmselfsup>=1.0.0rc3", to install mmselfsup
# import mmselfsup.models to trigger register_module in mmselfsup
custom_imports = dict(imports=['mmselfsup.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220927-e043f51a.pth'  # noqa
deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [512, 1024, 2048]

model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmselfsup.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3, 4), # Note: out_indices of ResNet in MMSelfSup are 1 larger than those in MMdet and MMCls
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=channels, # Note: The 3 channels of ResNet-50 output are [512, 1024, 2048], which do not match the original yolov5-s neck and need to be changed.
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels, # input channels of head need to be changed accordingly
            widen_factor=widen_factor))
)
```

### Don't used pre-training weights

When we replace the backbone network, the model initialization is trained by default loading the pre-training weight of the backbone network. Instead of using the pre-training weights of the backbone network, if you want to train the time model from scratch,
You can set `init_cfg` in 'backbone' to 'None'. In this case, the backbone network will be initialized with the default initialization method, instead of using the trained pre-training weight.

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [512, 1024, 2048]

model = dict(
   backbone=dict(
       _delete_=True, # Delete the backbone field in _base_
       type='mmdet.ResNet', # Using ResNet from mmdet
       depth=50,
       num_stages=4,
       out_indices=(1, 2, 3),
       frozen_stages=1,
       norm_cfg=dict(type='BN', requires_grad=True),
       norm_eval=True,
       style='pytorch',
       init_cfg=None # If init_cfg is set to None, backbone will not be initialized with pre-trained weights
   ),
   neck=dict(
       type='YOLOv5PAFPN',
       widen_factor=widen_factor,
       in_channels=channels, # Note: The 3 channels of ResNet-50 output are [512, 1024, 2048], which do not match the original yolov5-s neck and need to be changed.
       out_channels=channels),
   bbox_head=dict(
       type='YOLOv5Head',
       head_module=dict(
           type='YOLOv5HeadModule',
           in_channels=channels, # input channels of head need to be changed accordingly
           widen_factor=widen_factor))
)
```
