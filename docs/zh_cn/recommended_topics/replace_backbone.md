# 轻松更换主干网络

```{note}
1. 使用其他主干网络时，你需要保证主干网络的输出通道与 Neck 的输入通道相匹配。
2. 下面给出的配置文件，仅能确保训练可以正确运行，直接训练性能可能不是最优的。因为某些 backbone 需要配套特定的学习率、优化器等超参数。后续会在“训练技巧章节”补充训练调优相关内容。
```

## 使用 MMYOLO 中注册的主干网络

假设想将 `YOLOv6EfficientRep`  作为 `YOLOv5` 的主干网络，则配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    backbone=dict(
        type='YOLOv6EfficientRep',
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True))
)
```

## 跨库使用主干网络

OpenMMLab 2.0 体系中 MMYOLO、MMDetection、MMClassification、MMSelfsup 中的模型注册表都继承自 MMEngine 中的根注册表，允许这些 OpenMMLab 开源库直接使用彼此已经实现的模块。 因此用户可以在 MMYOLO 中使用来自 MMDetection、MMClassification、MMSelfsup 的主干网络，而无需重新实现。

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

由于 MMClassification 提供了 Py**T**orch **Im**age **M**odels (`timm`) 主干网络的封装，用户也可以通过 MMClassification 直接使用 `timm` 中的主干网络。假设想将 `EfficientNet-B1`作为 `YOLOv5` 的主干网络，则配置文件如下：

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

### 使用在 MMSelfSup 中实现的主干网络

假设想将 MMSelfSup 中 `MoCo v3`  自监督训练的 `ResNet-50` 作为 `YOLOv5` 的主干网络，则配置文件如下：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# 请先使用命令： mim install "mmselfsup>=1.0.0rc3"，安装 mmselfsup
# 导入 mmselfsup.models 使得可以调用 mmselfsup 中注册的模块
custom_imports = dict(imports=['mmselfsup.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220927-e043f51a.pth'  # noqa
deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [512, 1024, 2048]

model = dict(
    backbone=dict(
        _delete_=True, # 将 _base_ 中关于 backbone 的字段删除
        type='mmselfsup.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3, 4), # 注意：MMSelfSup 中 ResNet 的 out_indices 比 MMdet 和 MMCls 的要大 1
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
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

### 不使用预训练权重

通常情况下，骨干网络初始化都是优先选择预训练权重。如果你不想使用预训练权重，而是想从头开始训练时模型时，
我们可以将 `backbone` 中的 `init_cfg` 设置为 `None`，此时骨干网络将会以默认的初始化方法进行初始化，
而不会使用训练好的预训练权重进行初始。以下是以 `YOLOv5` 使用 resnet 作为主干网络为例子，其余算法也是同样的处理：

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
        init_cfg=None # init_cfg 设置为 None，则 backbone 将不会使用预训练好的权重进行初始化了
    ),
    neck=dict(
        type='YOLOv5PAFPN',
        widen_factor=widen_factor,
        in_channels=channels, # 注意：ResNet-50 输出的 3 个通道是 [512, 1024, 2048]，和原先的 yolov5-s neck 不匹配，需要更改
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels, # head 部分输入通道也要做相应更改
            widen_factor=widen_factor))
)
```
