# How to xxx

本教程收集了任何如何使用 MMYOLO 进行 xxx 的答案。 如果您遇到有关`如何做`的问题及答案，请随时更新此文档！

## 给主干网络增加插件

[更多的插件使用](docs/zh_cn/advanced_guides/plugins.md)

## 应用多个 Neck

如果你想堆叠多个 Neck，可以直接在配置文件中的 Neck 参数，MMYOLO 支持以 `List` 形式拼接多个 Neck 配置，你需要保证上一个 Neck 的输出通道与下一个 Neck
的输入通道相匹配。如需要调整通道，可以插入 `mmdet.ChannelMapper` 模块用来对齐多个 Neck 之间的通道数量。具体配置如下：

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
            out_channels=[256, 512, 1024],
            # 因为 out_channels 由 widen_factor 控制，YOLOv5PAFPN 的 out_channels = out_channels * widen_factor
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
    ],
    bbox_head=dict(head_module=dict(in_channels=[512, 512, 512]))
    # 因为 out_channels 由 widen_factor 控制，YOLOv5HeadModuled 的 in_channels * widen_factor 才会等于最后一个 neck 的 out_channels
)
```

## 跨库使用主干网络

OpenMMLab 2.0 体系中 MMYOLO、MMDetection、MMClassification、MMSegmentation 中的模型注册表都继承自 MMEngine 中的根注册表，允许这些 OpenMMLab
开源库直接使用彼此已经实现的模块。 因此用户可以在 MMYOLO 中使用来自 MMDetection、MMClassification 的主干网络，而无需重新实现。

```{note}
1. 使用其他主干网络时，你需要保证主干网络的输出通道与 Neck 的输入通道相匹配。
2. 下面给出的配置文件，仅能确保训练可以正确运行，直接训练性能可能不是最优的。因为某些 backbone 需要配套特定的学习率、优化器等超参数。后续会在“训练技巧章节”补充训练调优相关内容。
```

### 使用在 MMDetection 中实现的主干网络

1. 假设想将 `ResNet-50` 作为 `YOLOv5` 的主干网络，则配置文件如下：

   ```python
   _base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

   deepen_factor = _base_.deepen_factor
   widen_factor = 1.0
   channels = [512, 1024, 2048]

   model = dict(
       backbone=dict(
           _delete_=True, # 将 _base_ 中关于 backbone 的字段删除
           type='mmdet.ResNet', # 使用 mmdet 中的 ResNet
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
           in_channels=channels, # 注意：ResNet-50 输出的3个通道是 [512, 1024, 2048]，和原先的 yolov5-s neck 不匹配，需要更改
           out_channels=channels),
       bbox_head=dict(
           type='YOLOv5Head',
           head_module=dict(
               type='YOLOv5HeadModule',
               in_channels=channels, # head 部分输入通道也要做相应更改
               widen_factor=widen_factor))
   )
   ```

2. 假设想将 `SwinTransformer-Tiny` 作为 `YOLOv5` 的主干网络，则配置文件如下：

   ```python
   _base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

   deepen_factor = _base_.deepen_factor
   widen_factor = 1.0
   channels = [192, 384, 768]
   checkpoint_file = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

   model = dict(
       backbone=dict(
           _delete_=True, # 将 _base_ 中关于 backbone 的字段删除
           type='mmdet.SwinTransformer', # 使用 mmdet 中的 SwinTransformer
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
           in_channels=channels, # 注意：SwinTransformer-Tiny 输出的3个通道是 [192, 384, 768]，和原先的 yolov5-s neck 不匹配，需要更改
           out_channels=channels),
       bbox_head=dict(
           type='YOLOv5Head',
           head_module=dict(
               type='YOLOv5HeadModule',
               in_channels=channels, # head 部分输入通道也要做相应更改
               widen_factor=widen_factor))
   )
   ```

### 使用在 MMClassification 中实现的主干网络

1. 假设想将 `ConvNeXt-Tiny` 作为 `YOLOv5` 的主干网络，则配置文件如下：

   ```python
   _base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

   # 请先使用命令： mim install "mmcls>=1.0.0rc2"，安装 mmcls
   # 导入 mmcls.models 使得可以调用 mmcls 中注册的模块
   custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
   checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa
   deepen_factor = _base_.deepen_factor
   widen_factor = 1.0
   channels = [192, 384, 768]

   model = dict(
       backbone=dict(
           _delete_=True, # 将 _base_ 中关于 backbone 的字段删除
           type='mmcls.ConvNeXt', # 使用 mmcls 中的 ConvNeXt
           arch='tiny',
           out_indices=(1, 2, 3),
           drop_path_rate=0.4,
           layer_scale_init_value=1.0,
           gap_before_final_norm=False,
           init_cfg=dict(
               type='Pretrained', checkpoint=checkpoint_file,
               prefix='backbone.')), # MMCls 中主干网络的预训练权重含义 prefix='backbone.'，为了正常加载权重，需要把这个 prefix 去掉。
       neck=dict(
           type='YOLOv5PAFPN',
           deepen_factor=deepen_factor,
           widen_factor=widen_factor,
           in_channels=channels, # 注意：ConvNeXt-Tiny 输出的3个通道是 [192, 384, 768]，和原先的 yolov5-s neck 不匹配，需要更改
           out_channels=channels),
       bbox_head=dict(
           type='YOLOv5Head',
           head_module=dict(
               type='YOLOv5HeadModule',
               in_channels=channels, # head 部分输入通道也要做相应更改
               widen_factor=widen_factor))
   )
   ```

2. 假设想将 `MobileNetV3-small` 作为 `YOLOv5` 的主干网络，则配置文件如下：

   ```python
   _base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

   # 请先使用命令： mim install "mmcls>=1.0.0rc2"，安装 mmcls
   # 导入 mmcls.models 使得可以调用 mmcls 中注册的模块
   custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
   checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_small-8427ecf0.pth'  # noqa
   deepen_factor = _base_.deepen_factor
   widen_factor = 1.0
   channels = [24, 48, 96]

   model = dict(
       backbone=dict(
           _delete_=True, # 将 _base_ 中关于 backbone 的字段删除
           type='mmcls.MobileNetV3', # 使用 mmcls 中的 MobileNetV3
           arch='small',
           out_indices=(3, 8, 11), # 修改 out_indices
           init_cfg=dict(
               type='Pretrained',
               checkpoint=checkpoint_file,
               prefix='backbone.')), # MMCls 中主干网络的预训练权重含义 prefix='backbone.'，为了正常加载权重，需要把这个 prefix 去掉。
       neck=dict(
           type='YOLOv5PAFPN',
           deepen_factor=deepen_factor,
           widen_factor=widen_factor,
           in_channels=channels, # 注意：MobileNetV3-small 输出的3个通道是 [24, 48, 96]，和原先的 yolov5-s neck 不匹配，需要更改
           out_channels=channels),
       bbox_head=dict(
           type='YOLOv5Head',
           head_module=dict(
               type='YOLOv5HeadModule',
               in_channels=channels, # head 部分输入通道也要做相应更改
               widen_factor=widen_factor))
   )
   ```

### 通过 MMClassification 使用 `timm` 中实现的主干网络

由于 MMClassification 提供了 Py**T**orch **Im**age **M**odels (`timm`) 主干网络的封装，用户也可以通过 MMClassification 直接使用 `timm`
中的主干网络。假设想将 `EfficientNet-B1`作为 `YOLOv5` 的主干网络，则配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# 请先使用命令： mim install "mmcls>=1.0.0rc2"，安装 mmcls
# 以及： pip install timm，安装 timm
# 导入 mmcls.models 使得可以调用 mmcls 中注册的模块
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)

deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [40, 112, 320]

model = dict(
    backbone=dict(
        _delete_=True,  # 将 _base_ 中关于 backbone 的字段删除
        type='mmcls.TIMMBackbone',  # 使用 mmcls 中的 timm 主干网络
        model_name='efficientnet_b1',  # 使用 TIMM 中的 efficientnet_b1
        features_only=True,
        pretrained=True,
        out_indices=(2, 3, 4)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=channels,  # 注意：EfficientNet-B1 输出的3个通道是 [40, 112, 320]，和原先的 yolov5-s neck 不匹配，需要更改
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels,  # head 部分输入通道也要做相应更改
            widen_factor=widen_factor))
)
```
