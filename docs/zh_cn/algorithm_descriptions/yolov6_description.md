# YOLOv6 原理和实现全解析

## 0 简介

以上结构图 xxx 绘制。

YOLOv6 有一系列适用于各种工业场景的模型，包括N/T/S/M/L，考虑到模型的大小，其架构有所不同，以获得更好的精度-速度权衡。 此外，还引入了一些 "Bag-of-freebies "方法来进一步提高性能，如自我渐变和更多的训练周期。 在工业部署方面，我们采用QAT与信道蒸馏和图形优化来追求极端的性能（后续支持）。

简单来说 YOLOv6 开源库的主要特点为：

1. 统一设计了更高效的 Backbone 和 Neck：受到硬件感知神经网络设计思想的启发，基于 RepVGG style 设计了可重参数化、更高效的骨干网络 EfficientRep Backbone 和 Rep-PAN Neck。
2. 相比于 YOLOX 的 Decoupled Head，进一步优化设计了简洁有效的 Efficient Decoupled Head，在维持精度的同时，降低了一般解耦头带来的额外延时开销。
3. 在训练策略上，采用 Anchor-free 的策略，同时辅以 SimOTA 标签分配策略以及 SIoU 边界框回归损失来进一步提高检测精度。

本文将从 YOLOv6 算法本身原理讲起，然后重点分析 MMYOLO 中的实现。关于 YOLOv6 的使用指南和速度等对比请阅读本文的后续内容。

希望本文能够成为你入门和掌握 YOLOv6 的核心文档。由于 YOLOv6 本身也在不断迭代更新，我们也会不断的更新本文档。请注意阅读最新版本。

MMYOLO 实现配置：https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov6/

YOLOv6 官方开源库地址：https://github.com/meituan/YOLOv6

## 1 YLOLv6 2.0 算法原理和 MMYOLO 实现解析

YOLOv6 2.0 官方 release 地址：https://github.com/meituan/YOLOv6/releases/tag/0.2.0

<div align=center >
<img alt="YOLOv6精度图" src="https://github.com/meituan/YOLOv6/blob/main/assets/speed_comparision_v2.png"/>
</div>

<div align=center >
<img alt="YOLOv6精度速度图" src="https://user-images.githubusercontent.com/25873202/201611723-0d3d02be-d778-4bdd-8010-fbcb9df8740e.png"/>
</div>

YOLOv6 和 YOLOv5 一样也可以分成数据增强、模型结构、loss 计算等组件，如下所示：

<div align=center >
<img alt="训练测试策略" src="https://user-images.githubusercontent.com/40284075/190542423-f6b20d8e-c82a-4a34-9065-c161c5e29e7c.png"/>
</div>

下面将从原理和结合 MMYOLO 的具体实现方面进行简要分析。

### 1.1 数据增强模块

YOLOv6 目标检测算法中使用的数据增强与 YOLOv5 基本一致，唯独不一样的是没有使用 Albu 的数据增强方式：

- **Mosaic 马赛克**
- **RandomAffine 随机仿射变换**
- **MixUp**
- ~~**图像模糊等采用 Albu 库实现的变换**~~
- **HSV 颜色空间增强**
- **随机水平翻转**

关于每一个增强的详细解释，详情请看 \[YOLOv5 数据增强模块\](docs/zh_cn/algorithm_descriptions/yolov5_description.md#1.1 数据增强模块)

另外，YOLOv6 参考了 YOLOX 的数据增强方式，分为 2 钟增强方法组，一开始和 YOLOv5 一致，但是在最后 15 个 epoch 的时候将 `Mosaic` 使用 `YOLOv5KeepRatioResize` + `LetterResize` 替代了，个人感觉是为了拟合真实情况。

### 1.2 网络结构

#### 1.2.1 Backbone

#### 1.2.2 Neck

#### 1.2.3 Head

### 1.3 正负样本匹配策略

#### 1.3.1 Anchor 设置

YOLOv6 采用与 YOLOX 一样的 Anchor-free 无锚范式，省略的了聚类和繁琐的Anchor超参设定，泛化能力强，解码逻辑简单。在训练的过程中会根据 feature size 去自动生成先验框。

#### 1.3.2 Bbox 编解码过程

- Bbox 解码过程
  YOLOv6 位置预测参数格式为 `xywh`, 表示中心点位置偏移量以及宽高预测参数。
  对于 `xy` 来说，预测的是 gt bbox 中心落在的网格中相对于该网格左上角的偏移量，`hw` 是预测的宽高，最后乘以 stride，变换原图尺寸进行计算。

- Bbox 编码过程
  编码过程与解码过程相反，得到 anchor 对应的 gt 后，求 gt 中心点的与 grid 左上角的偏移以及宽高

#### 1.3.3 匹配策略

### 1.4 Loss 设计

- Classes loss：使用的是 `mmdet.VarifocalLoss`
- Objectness loss：使用的是 `mmdet.CrossEntropyLoss`
- BBox loss：l/m/s使用的是 GIoULoss,  t/n 用的是 SIoULoss

另外 YOLOv6 在计算 loss 之前，根据 epoch 的不同，会经过不同的 Assigner：

- epoch \< 4，使用 `BatchATSSAssigner`
- epoch >= 4，使用 `BatchTaskAlignedAssigner`

### 1.5 优化策略和训练过程

#### 1.5.1 优化器分组

与 YOLOv5 一致，详情请看 [YOLOv5 优化器分组](yolov5_description.md#1.5.1 优化器分组)

#### 1.5.2 weight decay 参数自适应

与 YOLOv5 一致，详情请看 [YOLOv5 weight decay 参数自适应](./yolov5_description.md#1.5.2 weight decay 参数自适应)

### 1.6 推理和后处理过程

YOLOv6 后处理过程和 YOLOv5 高度类似，实际上 YOLO 系列的后处理逻辑都是类似的。
详情请看 [YOLOv5 推理和后处理过程](./yolov5_description.md#1.6 推理和后处理过程)

## 2 总结

本文对 YOLOv6 原理和在 MMYOLO 实现进行了详细解析，希望能帮助用户理解算法实现过程。同时请注意：由于 YOLOv6 本身也在不断更新，本开源库也会不断迭代，请及时阅读和同步最新版本。
