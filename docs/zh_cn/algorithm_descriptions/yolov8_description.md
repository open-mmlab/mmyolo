# YOLOv8 原理和实现全解析

## 0 简介

<div align=center >
<img alt="YOLOv8-P5_structure" src="https://user-images.githubusercontent.com/27466624/211974251-8de633c8-090c-47c9-ba52-4941dc9e3a48.jpg"/>
图 1：YOLOv8-P5 模型结构
</div>

以上结构图由 RangeKing@github 绘制。

YOLOv8 是 ultralytics 公司在 2023 年 1月 10 号开源的 YOLOv5 的下一个重大更新版本，目前支持图像分类、物体检测和实例分割任务，在还没有开源时就收到了用户的广泛关注。

按照官方描述，YOLOv8 是一个 SOTA 模型，它建立在以前 YOLO 版本的成功基础上，并引入了新的功能和改进，以进一步提升性能和灵活性。具体创新包括一个新的骨干网络、一个新的 Ancher-Free 检测头和一个新的损失函数，可以在从 CPU 到 GPU 的各种硬件平台上运行。
不过 ultralytics 并没有直接将开源库命名为 YOLOv8，而是直接使用 ultralytics 这个词，原因是 ultralytics 将这个库定位为算法框架，而非某一个特定算法，一个主要特点是可扩展性。其希望这个库不仅仅能够用于 YOLO 系列模型，而是能够支持非 YOLO 模型以及分类分割姿态估计等各类任务。
总而言之，ultralytics 开源库的两个主要优点是：

- 融合众多当前 SOTA 技术于一体
- 未来将支持其他 YOLO 系列以及 YOLO 之外的更多算法

<div align=center >
<img alt="YOLOv8-table" src="https://user-images.githubusercontent.com/17425982/212007736-f592bc70-3959-4ff6-baf7-a93c7ad1d882.png"/>
图 2：YOLOv8 性能曲线
</div>

下表为官方在 COCO Val 2017 数据集上测试的 mAP、参数量和 FLOPs 结果。可以看出 YOLOv8 相比 YOLOv5 精度提升非常多，但是 N/S/M 模型相应的参数量和 FLOPs 都增加了不少，从上图也可以看出相比 YOLOV5 大部分模型推理速度变慢了。

| **模型** | **YOLOv5**  | **params****(M)** | **FLOPs****@640 (B)** | **YOLOv8**  | **params****(M)** | **FLOPs****@640 (B)** |
| -------- | ----------- | ----------------- | --------------------- | ----------- | ----------------- | --------------------- |
| n        | 28.0(300e)  | 1.9               | 4.5                   | 37.3 (500e) | 3.2               | 8.7                   |
| s        | 37.4 (300e) | 7.2               | 16.5                  | 44.9 (500e) | 11.2              | 28.6                  |
| m        | 45.4 (300e) | 21.2              | 49.0                  | 50.2 (500e) | 25.9              | 78.9                  |
| l        | 49.0 (300e) | 46.5              | 109.1                 | 52.9 (500e) | 43.7              | 165.2                 |
| x        | 50.7 (300e) | 86.7              | 205.7                 | 53.9 (500e) | 68.2              | 257.8                 |

额外提一句，现在各个 YOLO 系列改进算法都在 COCO 上面有明显性能提升，但是在自定义数据集上面的泛化性还没有得到广泛验证，至今依然听到不少关于 YOLOv5 泛化性能较优异的说法。对各系列 YOLO 泛化性验证也是 MMYOLO 中一个特别关心和重点发力的方向。

阅读本文前，如果你对 YOLOv5、YOLOv6 和 RTMDet 不熟悉，可以先看下如下文档：
1. YOLOv5 原理和实现全解析 https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/yolov5_description.html
2. YOLOv6 原理和实现全解析 https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/yolov6_description.html
3. RTMDet 原理和实现全解析 https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/rtmdet_description.html

## 1 YOLOv8 概述

YOLOv8 算法的核心特性和改动可以归结为如下：

1. 提供了一个全新的 SOTA 模型，包括 P5 640 和 P6 1280 分辨率的目标检测网络和基于 YOLACT 的实例分割模型。和 YOLOv5 一样，基于缩放系数也提供了 N/S/M/L/X 尺度的不同大小模型，用于满足不同场景需求
2. 骨干网络和 Neck 部分可能参考了 YOLOv7 ELAN 设计思想，将 YOLOv5 的 C3 结构换成了梯度流更丰富的 C2f 结构，并对不同尺度模型调整了不同的通道数，属于对模型结构精心微调，不再是无脑一套参数应用所有模型，大幅提升了模型性能。不过这个 C2f 模块中存在 Split 等操作对特定硬件部署没有之前那么友好了
3. Head 部分相比 YOLOv5 改动较大，换成了目前主流的解耦头结构，将分类和检测头分离，同时也从 Anchor-Based 换成了 Anchor-Free
4. Loss 计算方面采用了 TaskAlignedAssigner 正样本分配策略，并引入了 Distribution Focal Loss 
5. 训练的数据增强部分引入了 YOLOX 中的最后 10 epoch 关闭 Mosiac 增强的操作，可以有效地提升精度

从上面可以看出，YOLOv8 主要参考了最近提出的诸如 YOLOX、YOLOv6、YOLOv7 和 PPYOLOE 等算法的相关设计，本身的创新点不多，偏向工程实践，主推的还是 ultralytics 这个框架本身。

下面将按照模型结构设计、Loss 计算、训练数据增强、训练策略和模型推理过程共 5 个部分详细介绍 YOLOv8 目标检测的各种改进，实例分割部分暂时不进行描述。

## 2 模型结构设计
模型完整图示可以看图 1。

在暂时不考虑 Head 情况下，对比 YOLOv5 和 YOLOv8 的 yaml 配置文件可以发现改动较小。

<div align=center >
<img alt="yaml" src="https://user-images.githubusercontent.com/17425982/212008977-28c3fc7b-ee00-4d56-b912-d77ded585d78.png"/>
图 3：YOLOv5 和 YOLOv8 YAML 文件对比
</div>

左侧为 YOLOv5-s，右侧为 YOLOv8-s

骨干网络和 Neck 的具体变化为：
- 第一个卷积层的 kernel 从 6x6 变成了 3x3 
- 所有的 C3 模块换成 C2f，结构如下所示，可以发现多了更多的跳层连接和额外的 Split 操作

<div align=center >
<img alt="module" src="https://user-images.githubusercontent.com/17425982/212009208-92f45c23-a024-49bb-a2ee-bb6f87adcc92.png"/>
图 4：YOLOv5 和 YOLOv8 模块对比
</div>

- 去掉了 Neck 模块中的 2 个卷积连接层
- Backbone 中 C2f 的block 数从 3-6-9-3 改成了 3-6-6-3
- 查看 N/S/M/L/X 等不同大小模型，可以发现 N/S 和 L/X 两组模型只是改了缩放系数，但是 S/M/L 等骨干网络的通道数设置不一样，没有遵循同一套缩放系数。如此设计的原因应该是同一套缩放系数下的通道设置不是最优设计，YOLOv7 网络设计时也没有遵循一套缩放系数作用于所有模型

Head 部分变化最大，从原先的耦合头变成了解耦头，并且从 YOLOv5 的 Anchor-Based 变成了 Anchor-Free。其结构如下所示：

<div align=center >
<img alt="head" src="https://user-images.githubusercontent.com/17425982/212009547-189e14aa-6f93-4af0-8446-adf604a46b95.png"/>
图 5：YOLOv8 Head 结果
</div>

可以看出，不再有之前的 objectness 分支，只有解耦的分类和回归分支，并且其回归分支使用了 Distribution Focal Loss 中提出的积分形式表示法。

##3 Loss 计算

Loss 计算过程包括 2 个部分： 正负样本分配策略和 Loss 计算。
现代目标检测器大部分都会在正负样本分配策略上面做文章，典型的如 YOLOX 的 simOTA、TOOD 的 TaskAlignedAssigner 和 RTMDet 的 DynamicSoftLabelAssigner，这类 Assigner 大都是动态分配策略，而 YOLOv5 采用的依然是静态分配策略。考虑到动态分配策略的优异性，YOLOv8 算法中则直接引用了 TOOD 的 TaskAlignedAssigner。
TaskAlignedAssigner 的匹配策略简单总结为： 根据分类与回归的分数加权的分数选择正样本。

```{math}
t=s^\alpha+u^\beta  \\
```

s 是标注类别对应的预测分值，u 是预测框和 gt 框的 iou，两者相乘就可以衡量对齐程度。
1. 对于每一个 GT，对所有的预测框基于 GT 类别对应分类分数，预测框与 GT 的 IoU  的加权得到一个关联分类以及回归的对齐分数 alignment_metrics
2. 对于每一个 GT，直接基于 alignment_metrics 对齐分数选取 topK 大的作为正样本

Loss 计算包括 2 个分支： 分类和回归分支，没有了之前的 objectness 分支。
- 分类分支依然采用 BCE Loss
- 回归分支需要和 Distribution Focal Loss 中提出的积分形式表示法绑定，因此使用了 Distribution Focal Loss， 同时还使用了 CIoU Loss
3 个 Loss 采用一定权重比例加权即可。

## 4 训练数据增强

数据增强方面和 YOLOv5 差距不大，只不过引入了 YOLOX 中提出的最后 10 个 epoch 关闭 Mosaic 的操作。假设训练 epoch 是 500，其示意图如下所示：
