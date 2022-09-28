# RTMDet

高性能，低延时的单阶段目标检测器

## 简介

最近一段时间，开源界涌现出了大量的高精度目标检测项目，其中最突出的就是 YOLO 系列，OpenMMLab 也在与社区的合作下推出了 MMYOLO。
在集成了如此多高性能高精度的学术界以及工业界的模型之后，RTMDet也对这些它们的结构
设计以及训练方式进行了经验性的总结，并进行了优化，推出了 RTMDe t高精度、低延时的
单阶段目标检测器。
**R**eal-**t**ime **M**odels for Object **Det**ection
(**R**elease **t**o **M**anufacture)

RTMDet 由 tiny/s/m/l/x 一系列不同大小的模型组成，为不同的应用场景提供了不同的选择。
其中，RTMDet-x 在 52.6 mAP 的精度下达到了300+FPS的推理速度。
而最轻量的模型 RTMDet-tiny，在仅有4M参数量的情况下也能够达到 40.9 mAP，且推理速度 \<1ms。

![img](https://user-images.githubusercontent.com/12907710/192182907-f9a671d6-89cb-4d73-abd8-c2b9dada3c66.png)

**注**:

1. 推理速度测试（不包含 NMS）是在 1 块 NVIDIA 3090 GPU 上的 `TensorRT 8.4.3, cuDNN 8.2.0, FP16, batch size=1` 条件里测试的。

## 2. 模型结构

## 整体结构

RTMDet 模型整体结构与目前主流的检测器 YOLO 系列相似。以 **L**arge 模型为例，整体由 `CSPNeXt` + `CSPNeXtPAFPN` + `共享卷积权重但分别计算 BN 的 Head` 构成。

### Backbone

`CSPNeXt` 整体以 `CSPDarknet` 为基础，共 5 层结构，包含 1 个 `Stem Layer` 和 4 个 `Stage Layer`：

- `Stem Layer` 是 3 层 3x3 kernel 的 `ConvModule` ，不同于之前的 `Focus` 模块或者 1 层 6x6 kernel 的 `ConvModule` 。
- `Stage Layer` 总体结构与已有模型类似，前 3 个 `Stage Layer` 由 1 个 `ConvModule` 和 1 个 `CSPLayer`  组成。第 4 个 `Stage Layer` 在 `ConvModule`  和  `CSPLayer` 中间增加了 `SPPF` 模块（MMDetection 版本为 `SPP` 模块）。
- 如上图 Details 部分，其中 `ConvModule` 为 3x3 `Conv2d` + `BatchNorm` + `SiLU 激活函数`。`CSPLayer` 由 3 个 `ConvModule` + n 个 `CSPNeXt Block`(带残差连接) + `Channel Attention` 组成。`CSPNeXt Block` 和 `Channel Attention` 模块在下节详细讲述。

#### CSPNeXt Block

RTMDet 对 `CSPDarknet` 的 `Basic Block` 进行了改进。
`DarkNet` （图 a）使用 1x1 与 3x3 卷积的 `Basic Block`。YOLOv6、v7、PPYOLO-E（图 b & c）使用了重参数化 Block。但重参数化的训练代价高，且不易量化，需要其他方式来弥补量化误差。
RTMDet 则借鉴了最近比较热门的 [ConvNeXt](https://arxiv.org/abs/2201.03545)、[RepLKNet](https://arxiv.org/abs/2203.06717) 的做法，为 `Basic Block` 加入了大 kernel 的 `depth-wise` 卷积（图 d），并将其命名为 `CSPNeXt Block`。

关于不同 kernel 大小的实验结果，如下表所示。

| Kernel  size | params     | flops     | latency-bs1-TRT-FP16 / ms | mAP      |
| ------------ | ---------- | --------- | ------------------------- | -------- |
| 3x3          | 50.8       | 79.61G    | 2.1                       | 50.0     |
| **5x5**      | **50.92M** | **79.7G** | **2.11**                  | **50.9** |
| 7x7          | 51.1       | 80.34G    | 2.73                      | 51.1     |

#### 调整检测器不同 stage 间的 block 数

由于 ` CSPNeXt Block` 内使用了 `depth-wise` 卷积，单个 block 内的层数增多，如果保持原有的 stage 内的 block 数，则会导致模型的推理速度大幅降低。

RTMDet 重新调整了不同 stage 间的 block 数，并调整了通道的超参，在保证了精度的情况下提升了推理速度。

关于不同 block 数的实验结果，如下表所示。

| Num  blocks                        | params    | flops     | latency-bs1-TRT-FP16 / ms | mAP      |
| ---------------------------------- | --------- | --------- | ------------------------- | -------- |
| L+3-9-9-3                          | 53.4      | 86.28     | 2.6                       | 51.4     |
| L+3-6-6-3                          | 50.92M    | 79.7G     | 2.11                      | 50.9     |
| **L+3-6-6-3  + channel attention** | **52.3M** | **79.9G** | **2.4**                   | **51.3** |

### Neck

#### Backbone 与 Neck 之间的参数量和计算量的均衡

[EfficientDet](https://arxiv.org/abs/1911.09070)、[NASFPN](https://arxiv.org/abs/1904.07392) 等工作在改进 Neck 时往往聚焦于如何修改特征融合的方式。
但引入过多的连接会增加检测器的延时，并增加内存开销。

所以 RTMDet 选择不引入额外的连接，而是改变 Backbone 与 Neck 间参数量的配比。
实验发现，当 Neck 在整个模型中的参数量占比更高时，延时更低，且对精度的影响很小。作者在直播答疑时回复，RTMDet 在 Neck 这一部分的实验参考了 [GiraffeDet](https://arxiv.org/abs/2202.04256) 的做法，但没有像 GiraffeDet 一样引入额外连接（详细可参见 [RTMDet 发布视频](https://www.bilibili.com/video/BV1e841147GD) 31分40秒）。

关于不同参数量配比的实验结果，如下表所示。

| Model  size | Backbone | Neck    | params     | flops      | latency  / ms | mAP      |
| ----------- | -------- | ------- | ---------- | ---------- | ------------- | -------- |
| **S**       | **47%**  | **45%** | **8.54M**  | **15.76G** | **1.21**      | **43.9** |
| S           | 63%      | 29%     | 9.01M      | 15.85G     | 1.37          | 43.7     |
| **L**       | **47%**  | **45%** | **50.92M** | **79.7G**  | **2.11**      | **50.9** |
| L           | 63%      | 29%     | 57.43M     | 93.73      | 2.57          | 51.0     |

**注**：MMYOLO 和 MMDetection 中 Neck 的具体实现和连接有细微不同。

### Head

传统的 YOLO 系列都使用同一 Head 进行分类和回归。YOLOX 则将分类和回归分支解耦，PPYOLO-E 和 YOLOv6 则引入了 TOOD 中的结构。它们在不同特征层级之间都使用独立的 Head，因此 Head 在模型中也占有较多的参数量。

RTMDet 参考了 NAS-FPN 中的做法，使用了 `SepBNHead`，在不同层之间共享卷积权重，但是独立计算 BN 的统计量。

关于不同结构 Head 的实验结果，如下表所示。

| Head  type         | params    | flops     | latency  / ms | mAP      |
| ------------------ | --------- | --------- | ------------- | -------- |
| Fully-shared  head | 52.32     | 80.23     | 2.44          | 48.0     |
| Separated  head    | 57.03     | 80.23     | 2.44          | 51.2     |
| **SepBN** **head** | **52.32** | **80.23** | **2.44**      | **51.3** |

同时，RTMDet 也延续了作者之前在 [NanoDet](https://zhuanlan.zhihu.com/p/306530300) 中的思想，使用 [Quality Focal Loss](https://arxiv.org/abs/2011.12885)，去掉 Objectness 分支并去掉 Objectness 分支，进一步将 Head 轻量化。
