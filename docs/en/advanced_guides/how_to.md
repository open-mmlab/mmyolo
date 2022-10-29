This tutorial collects answers to any `How to xxx with MMYOLO`. Feel free to update this doc if you meet new questions about `How to` and find the answers!

# Add plugins to the Backbone network

MMYOLO supports adding plugins such as none_local and dropout after different stages of Backbone. Users can directly manage plugins by modifying the plugins parameter of backbone in config. For example, add GeneralizedAttention plugins for `YOLOv5`. The configuration files are as follows:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type='mmdet.GeneralizedAttention',
                    spatial_range=-1,
                    num_heads=8,
                    attention_type='0011',
                    kv_stride=2),
                stages=(False, False, True, True)),
        ], ))
```

`cfg` parameter indicates the specific configuration of the plug-in. The `stages` parameter indicates whether to add plug-ins after the corresponding stage of the backbone. The length of list `stages` must be the same as the number of backbone stages.

## Apply multiple Necks

If you want to stack multiple Necks, you can directly set the Neck parameters in the config. MMYOLO supports concatenating multiple Necks in the form of `List`. You need to ensure that the output channel of the previous Neck matches the input channel of the next Neck. If you need to adjust the number of channels, you can insert the `mmdet.ChannelMapper` module to align the number of channels between multiple Necks. The specific configuration is as follows:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

deepen_factor = _base_.deepen_factor
widen_factor = _base_.widen_factor
model = dict(
    type='YOLODetector',
    neck=[
        dict(
            type='YOLOv5PAFPN',
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            in_channels=[256, 512, 1024],
            out_channels=[256, 512, 1024], # The out_channels is controlled by widen_factor，so the YOLOv5PAFPN's out_channels equls to out_channels * widen_factor
            num_csp_blocks=3,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        dict(
            type='mmdet.ChannelMapper',
            in_channels=[128, 256, 512],
            out_channels=128,
        ),
        dict(
            type='mmdet.DyHead',
            in_channels=128,
            out_channels=256,
            num_blocks=2,
            # disable zero_init_offset to follow official implementation
            zero_init_offset=False)
    ]
    bbox_head=dict(head_module=dict(in_channels=[512,512,512])) # The out_channels is controlled by widen_factor，so the YOLOv5HeadModuled in_channels * widen_factor equals to  the last neck's out_channels
)
```

## Use backbone network implemented in other OpenMMLab repositories

The model registry in MMYOLO, MMDetection, MMClassification, and MMSegmentation all inherit from the root registry in MMEngine in the OpenMMLab 2.0 system, allowing these repositories to directly use modules already implemented by each other. Therefore, in MMYOLO, users can use backbone networks from MMDetection and MMClassification without reimplementation.

```{note}
When using other backbone networks, you need to ensure that the output channels of the backbone network match the input channels of the neck network.
```

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
           in_channels=channels,
           out_channels=channels),
       bbox_head=dict(
           type='YOLOv5Head',
           head_module=dict(
               type='YOLOv5HeadModule',
               in_channels=channels,
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
           in_channels=channels,
           out_channels=channels),
       bbox_head=dict(
           type='YOLOv5Head',
           head_module=dict(
               type='YOLOv5HeadModule',
               in_channels=channels,
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
           in_channels=channels,
           out_channels=channels),
       bbox_head=dict(
           type='YOLOv5Head',
           head_module=dict(
               type='YOLOv5HeadModule',
               in_channels=channels,
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
           in_channels=channels,
           out_channels=channels),
       bbox_head=dict(
           type='YOLOv5Head',
           head_module=dict(
               type='YOLOv5HeadModule',
               in_channels=channels,
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
        in_channels=channels,
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels,
            widen_factor=widen_factor))
)
```
