# How to xxx

本教程收集了任何如何使用 MMYOLO 进行 xxx 的答案。 如果您遇到有关`如何做`的问题及答案，请随时更新此文档！

## 给主干网络增加插件

[更多的插件使用](plugins.md)

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

## 更换主干网络

```{note}
1. 使用其他主干网络时，你需要保证主干网络的输出通道与 Neck 的输入通道相匹配。
2. 下面给出的配置文件，仅能确保训练可以正确运行，直接训练性能可能不是最优的。因为某些 backbone 需要配套特定的学习率、优化器等超参数。后续会在“训练技巧章节”补充训练调优相关内容。
```

### 使用 MMYOLO 中注册的主干网络

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

### 跨库使用主干网络

OpenMMLab 2.0 体系中 MMYOLO、MMDetection、MMClassification、MMSelfsup 中的模型注册表都继承自 MMEngine 中的根注册表，允许这些 OpenMMLab 开源库直接使用彼此已经实现的模块。 因此用户可以在 MMYOLO 中使用来自 MMDetection、MMClassification、MMSelfsup 的主干网络，而无需重新实现。

#### 使用在 MMDetection 中实现的主干网络

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

#### 使用在 MMClassification 中实现的主干网络

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

#### 通过 MMClassification 使用 `timm` 中实现的主干网络

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

#### 使用在 MMSelfSup 中实现的主干网络

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

#### 不使用预训练权重

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

#### 冻结 backbone 或 neck 的权重

在 MMYOLO 中我们可以通过设置 `frozen_stages` 参数去冻结主干网络的部分 `stage`, 使这些 `stage` 的参数不参与模型的更新。
需要注意的是：`frozen_stages = i` 表示的意思是指从最开始的 `stage` 开始到第 `i` 层 `stage` 的所有参数都会被冻结。下面是 `YOLOv5` 的例子，其他算法也是同样的逻辑：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    backbone=dict(
        frozen_stages=1 # 表示第一层 stage 以及它之前的所有 stage 中的参数都会被冻结
    ))
```

此外， MMYOLO 中也可以通过参数 `freeze_all` 去冻结整个 `neck` 的参数。下面是 `YOLOv5` 的例子，其他算法也是同样的逻辑：

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    neck=dict(
        freeze_all=True # freeze_all=True 时表示整个 neck 的参数都会被冻结
    ))
```

## 输出预测结果

如果想将预测结果保存为特定的文件，用于离线评估，目前 MMYOLO 支持 json 和 pkl 两种格式。

```{note}
json 文件仅保存 `image_id`、`bbox`、`score` 和 `category_id`； json 文件可以使用 json 库读取。
pkl 保存内容比 json 文件更多，还会保存预测图片的文件名和尺寸等一系列信息； pkl 文件可以使用 pickle 库读取。
```

### 输出为 json 文件

如果想将预测结果输出为 json 文件，则命令如下：

```shell
python tools/test.py ${CONFIG} ${CHECKPOINT} --json-prefix ${JSON_PREFIX}
```

`--json-prefix` 后的参数输入为文件名前缀（无需输入 `.json` 后缀），也可以包含路径。举一个具体例子：

```shell
python tools/test.py configs\yolov5\yolov5_s-v61_syncbn_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --json-prefix work_dirs/demo/json_demo
```

运行以上命令会在 `work_dirs/demo` 文件夹下，输出 `json_demo.bbox.json` 文件。

### 输出为 pkl 文件

如果想将预测结果输出为 pkl 文件，则命令如下：

```shell
python tools/test.py ${CONFIG} ${CHECKPOINT} --out ${OUTPUT_FILE} [--cfg-options ${OPTIONS [OPTIONS...]}]
```

`--out` 后的参数输入为完整文件名（**必须输入** `.pkl` 或 `.pickle` 后缀），也可以包含路径。举一个具体例子：

```shell
python tools/test.py configs\yolov5\yolov5_s-v61_syncbn_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --out work_dirs/demo/pkl_demo.pkl
```

运行以上命令会在 `work_dirs/demo` 文件夹下，输出 `pkl_demo.pkl` 文件。

## 使用 mim 跨库调用其他 OpenMMLab 仓库的脚本

```{note}
1. 目前暂不支持跨库调用所有脚本，正在修复中。等修复完成，本文档会添加更多的例子。
2. 绘制 mAP 和 计算平均训练速度 两项功能在 MMDetection dev-3.x 分支中修复，目前需要通过源码安装该分支才能成功调用。
```

### 日志分析

#### 曲线图绘制

MMDetection 中的 `tools/analysis_tools/analyze_logs.py` 可利用指定的训练 log 文件绘制 loss/mAP 曲线图， 第一次运行前请先运行 `pip install seaborn` 安装必要依赖。

```shell
mim run mmdet analyze_logs plot_curve \
    ${LOG} \                                     # 日志文件路径
    [--keys ${KEYS}] \                           # 需要绘制的指标，默认为 'bbox_mAP'
    [--start-epoch ${START_EPOCH}]               # 起始的 epoch，默认为 1
    [--eval-interval ${EVALUATION_INTERVAL}] \   # 评估间隔，默认为 1
    [--title ${TITLE}] \                         # 图片标题，无默认值
    [--legend ${LEGEND}] \                       # 图例，默认为 None
    [--backend ${BACKEND}] \                     # 绘制后端，默认为 None
    [--style ${STYLE}] \                         # 绘制风格，默认为 'dark'
    [--out ${OUT_FILE}]                          # 输出文件路径
# [] 代表可选参数，实际输入命令行时，不用输入 []
```

样例：

- 绘制分类损失曲线图

  ```shell
  mim run mmdet analyze_logs plot_curve \
      yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json \
      --keys loss_cls \
      --legend loss_cls
  ```

  <img src="https://user-images.githubusercontent.com/27466624/204747359-754555df-1f97-4d5c-87ca-9ad3a0badcce.png" width="600"/>

- 绘制分类损失、回归损失曲线图，保存图片为对应的 pdf 文件

  ```shell
  mim run mmdet analyze_logs plot_curve \
      yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json \
      --keys loss_cls loss_bbox \
      --legend loss_cls loss_bbox \
      --out losses_yolov5_s.pdf
  ```

  <img src="https://user-images.githubusercontent.com/27466624/204748560-2d17ce4b-fb5f-4732-a962-329109e73aad.png" width="600"/>

- 在同一图像中比较两次运行结果的 bbox mAP

  ```shell
  mim run mmdet analyze_logs plot_curve \
      yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json \
      yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739.log.json \
      --keys bbox_mAP \
      --legend yolov5_s yolov5_n \
      --eval-interval 10 # 注意评估间隔必须和训练时设置的一致，否则会报错
  ```

<img src="https://user-images.githubusercontent.com/27466624/204748704-21db9f9e-386e-449c-91c7-2ce3f8b51f24.png" width="600"/>

#### 计算平均训练速度

```shell
mim run mmdet analyze_logs cal_train_time \
    ${LOG} \                                # 日志文件路径
    [--include-outliers]                    # 计算时包含每个 epoch 的第一个数据
```

样例：

```shell
mim run mmdet analyze_logs cal_train_time \
    yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json
```

输出以如下形式展示：

```text
-----Analyze train time of yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json-----
slowest epoch 278, average time is 0.1705 s/iter
fastest epoch 300, average time is 0.1510 s/iter
time std over epochs is 0.0026
average iter time: 0.1556 s/iter
```

### 打印完整配置文件

MMDetection 中的 `tools/misc/print_config.py` 脚本可将所有配置继承关系展开，打印相应的完整配置文件。调用命令如下：

```shell
mim run mmdet print_config \
    ${CONFIG} \                              # 需要打印的配置文件路径
    [--save-path] \                          # 保存文件路径，必须以 .py, .json 或者 .yml 结尾
    [--cfg-options ${OPTIONS [OPTIONS...]}]  # 通过命令行参数修改配置文件
```

样例：

```shell
mim run mmdet print_config \
    configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py \
    --save-path ./work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon_whole.py
```

运行以上命令，会将 `yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py` 继承关系展开后的配置文件保存到 `./work_dirs` 文件夹内的 `yolov5_s-v61_syncbn_fast_1xb4-300e_balloon_whole.py` 文件中。

## 设置随机种子

如果想要在训练时指定随机种子，可以使用以下命令：

```shell
python ./tools/train.py \
    ${CONFIG} \                               # 配置文件路径
    --cfg-options randomness.seed=2023 \      # 设置随机种子为 2023
    [randomness.diff_rank_seed=True] \        # 根据 rank 来设置不同的种子。
    [randomness.deterministic=True]           # 把 cuDNN 后端确定性选项设置为 True
# [] 代表可选参数，实际输入命令行时，不用输入 []
```

`randomness` 有三个参数可设置，具体含义如下：

- `randomness.seed=2023` ，设置随机种子为 2023。

- `randomness.diff_rank_seed=True`，根据 rank 来设置不同的种子，`diff_rank_seed` 默认为 False。

- `randomness.deterministic=True`，把 cuDNN 后端确定性选项设置为 True，即把`torch.backends.cudnn.deterministic` 设为 True，把 `torch.backends.cudnn.benchmark` 设为False。`deterministic` 默认为 False。更多细节见 https://pytorch.org/docs/stable/notes/randomness.html。
