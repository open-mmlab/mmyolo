# 常见问题解答

我们在这里列出了使用时的一些常见问题及其相应的解决方案。 如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。 如果您无法在此获得帮助，请创建 [issue](https://github.com/open-mmlab/mmyolo/issues/new/choose) 提问，但是请在模板中填写所有必填信息，这有助于我们更快定位问题。

## 为什么要推出 MMYOLO？

为什么要推出 MMYOLO？ 为何要单独开一个仓库而不是直接放到 MMDetection 中？ 自从开源后，不断收到社区小伙伴们类似的疑问，答案可以归纳为以下三点：

**(1) 统一运行和推理平台**

目前目标检测领域出现了非常多 YOLO 的改进算法，并且非常受大家欢迎，但是这类算法基于不同框架不同后端实现，存在较大差异，缺少统一便捷的从训练到部署的公平评测流程。

**(2) 协议限制**

众所周知，YOLOv5 以及其衍生的 YOLOv6 和 YOLOv7 等算法都是 GPL 3.0 协议，不同于 MMDetection 的 Apache 协议。由于协议问题，无法将 MMYOLO 直接并入 MMDetection 中。

**(3) 多任务支持**

还有一层深远的原因： **MMYOLO 任务不局限于 MMDetection**，后续会支持更多任务例如基于 MMPose 实现关键点相关的应用，基于 MMTracking 实现追踪相关的应用，因此不太适合直接并入 MMDetection 中。

## projects 文件夹是用来干什么的？

projects 文件夹是 OpenMMLab 2.0 中引入的一个全新文件夹。其初衷有如下 3 点：

1. 便于社区贡献。由于 OpenMMLab 系列代码库对于代码合入有一套规范严谨的流程，这不可避免的会导致算法复现周期很长，不利于社区贡献
2. 便于快速支持新算法。算法开发周期过长同样会导致用户无法尽快体验最新算法
3. 便于快速支持新方向和新特性。新发展方向或者一些新的特性可能和现如今代码库中的设计有些不兼容，没法快速合入到代码库中

综上所述，projects 文件夹的引入主要是解决算法复现周期过长导致的新算法支持速度较慢，新特性支持较复杂等多个问题。 projects 中每个文件夹属于一个完全独立的工程，社区用户可以通过
projects 快速支持一些在当前版本中较难支持或者想快速支持的新算法和新特性。等后续设计稳定或者代码符合合入规范，则会考虑合入到主分支中。

## YOLOv5 backbone 替换为 Swin 后效果很差

在 [轻松更换主干网络](../recommended_topics/replace_backbone.md) 一文中我们提供了大量替换 backbone 的教程，但是该文档只是教用户如何替换 backbone，直接训练不一定能得到比较优异的结果。原因是
不同 backbone 所需要的训练超参是不一样的，以 Swin 和 YOLOv5 backbone 为例两者差异较大，Swin 属于 transformer 系列算法，而 YOLOv5 backbone 属于卷积系列算法，其训练的优化器、学习率以及其他超参差异较大。
如果强行将 Swin 作为 YOLOv5 backbone 且想取得不错的效果，需要同时调整诸多参数。

## MM 系列开源库中有很多组件，如何在 MMYOLO 中使用？

在 OpenMMLab 2.0 中对多个 MM 系列开源库之间的模块跨库调用功能进行增强。目前在 MMYOLO 中可以在配置文件中通过 `MM 算法库 A.模块名` 来之间调用 MM 算法库 A 中已经被注册的任意模块。 具体例子可以参考
[轻松更换主干网络](../recommended_topics/replace_backbone.md) 中使用在 MMClassification 中实现的主干网络章节，其他模块调用也是相同的用法。

## MMYOLO 中是否可以加入纯背景图片进行训练？

将纯背景图片加入训练大部分情况可以抑制误报率，是否将纯背景图片加入训练功能已经大部分数据集上支持了。以 `YOLOv5CocoDataset` 为例，核心控制参数是 `train_dataloader.dataset.filter_cfg.filter_empty_gt`，如果 `filter_empty_gt` 为 True 表示将纯背景图片过滤掉不加入训练，
反之将纯背景图片加入到训练中。 目前 MMYOLO 中大部分算法都是默认将纯背景图片加入训练中。

## MMYOLO 是否有计算模型推理 FPS 脚本？

MMYOLO 是基于 MMDet 3.x 来开发的，在 MMDet 3.x 中提供了计算模型推理 FPS 的脚本。 具体脚本为 [benchmark](https://github.com/open-mmlab/mmdetection/blob/3.x/tools/analysis_tools/benchmark.py)。我们推荐大家使用 mim 直接跨库启动 MMDet 中的脚本而不是直接复制到 MMYOLO 中。
关于如果通过 mim 启动 MMDet 中脚本，可以查看 [使用 mim 跨库调用其他 OpenMMLab 仓库的脚本](../common_usage/mim_usage.md)。

## MMDeploy 和 EasyDeploy 有啥区别？

MMDeploy 是由 OpenMMLab 中部署团队开发的针对 OpenMMLab 系列算法库提供部署支持的开源库，支持各种后端和自定义等等强大功能。 EasyDeploy 是由社区小伙伴提供的一个相比 MMDeploy 更加简单易用的部署 projects。
EasyDeploy 支持的功能目前没有 MMDeploy 多，但是使用上更加简单。 MMYOLO 中同时提供对 MMDeploy 和 EasyDeploy 的支持，用户可以根据自己需求选择。

## COCOMetric 中如何查看每个类的 AP

只需要在配置中设置 `test_evaluator.classwise` 为 True，或者在 test.py 运行时候增加 `--cfg-options test_evaluator.classwise=True` 即可。

## MMYOLO 中为何没有支持 MMDet 类似的自动学习率缩放功能？

原因是实验发现 YOLO 系列算法不是非常满足线性缩放功能。在多个数据集上验证发现会出现不基于 batch size 自动学习率缩放效果好于缩放的情形。因此暂时 MMYOLO 还没有支持自动学习率缩放功能。

## 自己训练的模型权重尺寸为啥比官方发布的大？

原因是用户自己训练的权重通常包括 `optimizer`、`ema_state_dict` 和 `message_hub` 等额外数据，这部分数据我们会在模型发布时候自动删掉，而用户直接基于框架跑的模型权重是全部保留的，所以用户自己训练的模型权重尺寸会比官方发布的大。
你可以使用 [publish_model.py](https://github.com/open-mmlab/mmyolo/blob/main/tools/misc/publish_model.py) 脚本删掉额外字段。

## RTMDet 为何训练所占显存比 YOLOv5 多很多？

训练显存较多的原因主要是 assigner 部分的差异。YOLOv5 采用的是非常简单且高效的 shape 匹配 assigner，而 RTMDet 中采用的是动态的全 batch 计算的 dynamic soft label assigner，其内部的 Cost 矩阵需要消耗比较多的显存，特别是当前 batch 中标注框过多时候。
后续我们会考虑解决这个问题。

## 修改一些代码后是否需要重新安装 MMYOLO

在不新增 py 代码情况下， 如果你遵循最佳实践，即使用 `mim install -v -e .` 安装的 MMYOLO，则对本地代码所作的任何修改都会生效，无需重新安装。但是如果你是新增了 py 文件然后在里面新增的代码，则依然需要重新安装即运行 `mim install -v -e .`。

## 如何使用多个 MMYOLO 版本进行开发

推荐你拥有多个 MMYOLO 工程文件夹，例如 mmyolo-v1, mmyolo-v2。 在使用不同版本 MMYOLO 时候，你可以在终端运行前设置

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```

使得当前环境生效。如果要使用环境中安装默认的 MMYOLO 而不是当前正在在使用的，可以删除出现上述命令或者通过如下命令重置

```shell
unset PYTHONPATH
```

## 训练中保存最好模型

用户可以通过在配置中设置 `default_hooks.checkpoint.save_best` 参数来选择根据什么指标来筛选最优模型。以 `COCO` 数据集检测任务为例，
则 `default_hooks.checkpoint.save_best` 可以选择输入的参数有:

1. `auto` 将会根据验证集中的第一个评价指标作为筛选条件。
2. `coco/bbox_mAP` 将会根据 `bbox_mAP` 作为筛选条件。
3. `coco/bbox_mAP_50` 将会根据 `bbox_mAP_50` 作为筛选条件。
4. `coco/bbox_mAP_75` 将会根据 `bbox_mAP_75` 作为筛选条件。
5. `coco/bbox_mAP_s` 将会根据 `bbox_mAP_s` 作为筛选条件。
6. `coco/bbox_mAP_m` 将会根据 `bbox_mAP_m` 作为筛选条件。
7. `coco/bbox_mAP_l` 将会根据 `bbox_mAP_l` 作为筛选条件。

此外用户还可以选择筛选的逻辑，通过设置配置中的 `default_hooks.checkpoint.rule` 来选择判断逻辑，如：`default_hooks.checkpoint.rule=greater` 表示指标越大越好。更详细的使用可以参考 [checkpoint_hook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py) 来修改

## 如何进行非正方形输入尺寸训练和测试 ?

在 YOLO 系列算法中默认配置基本上都是 640x640 或者 1280x1280 正方形尺度输入训练的。用户如果想进行非正方形尺度训练，你可以修改配置中 `image_scale` 参数，并将其他对应位置进行修改即可。用户可以参考我们提供的 [yolov5_s-v61_fast_1xb12-40e_608x352_cat.py](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov5/yolov5_s-v61_fast_1xb12-40e_608x352_cat.py) 配置。
