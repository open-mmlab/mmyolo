# Algorithm principles and implementation with YOLOv5

## 0 Introduction

<div align=center >
<img alt="YOLOv5_structure_v3.3" src="https://user-images.githubusercontent.com/27466624/192134657-a8d0286d-640c-445f-89bd-fda751094a4a.jpg"/>
</div>

RangeKing@github provides the graph above. Thanks, RangeKing!

YOLOv5 is an open-sourced object detection algorithm for real-time industrial applications which has received extensive attention. We believe that the reason for the explosion of YOLOv5 is not simply due to its excellent performance. It is more about the overall utility and robustness of its library.
In short, the main features of YOLOv5 are:

1. **friendly and perfect deployment supports**
2. **fast training speech**: the training time in the case of 300 epochs is similar to most of the one-stage and two-stage algorithms under 12 epochs, such as RetinaNet, ATSS, and Faster R-CNN.
3. **abundant optimization for corner cases**: YOLOv5 has implemented many optimizations. The functions and documentation are richer as well.

This article will start with the principle of the YOLOv5 algorithm and then focus on analyzing the implementation in MMYOLO. The follow-up part includes the guide and speed benchmark of YOLOv5.

We hope this article can become your core document to get started and master YOLOv5. Since YOLOv5 is still constantly updated, we will also keep updating this document. So please always catch up with the latest version.

MMYOLO implementation configuration: https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/

YOLOv5 official repository: https://github.com/ultralytics/yolov5

## 1 v6.1 algorithm principle and MMYOLO implementation analysis

YOLOv5 official release: https://github.com/ultralytics/yolov5/releases/tag/v6.1

<div align=center >
<img alt="YOLOv5 accuracy" src="https://user-images.githubusercontent.com/40284075/190542120-29d46b7e-ce3c-436a-9933-cfc9f86787bf.png"/>
</div>

<div align=center >
<img alt="YOLOv5精度速度图" src="https://user-images.githubusercontent.com/40284075/190542279-37734629-2b59-4bd8-a9bf-757875a93eed.png"/>
</div>

The performance is shown in the table above. YOLOv5 has two models with different scales. P6 is larger with a 1280x1280 input size, whereas P5 is the model which will be used more often. This article focuses on the structure of the P5 model.

Object detection algorithms can generally be divided into data augmentation, model structure, loss calculation, etc. It is the same as YOLOv5:

<div align=center >
<img alt="训练测试策略" src="https://user-images.githubusercontent.com/40284075/190542423-f6b20d8e-c82a-4a34-9065-c161c5e29e7c.png"/>
</div>

Now we will briefly analyze the principle and our specific implementation in MMYOLO.

### 1.1 Data augmentation

Many data augmentation methods are used in YOLOv5, including:

- **Mosaic**
- **RandomAffine**
- **MixUp**
- **Image blue and other transformations with Albu**
- **HSV color space enhancement**
- **Random horizontal flips**

The Mosaic data augmentation probability is 1, which means it will always be triggered. MixUp is not used for the small and nano models, and the probability is set to 0.1 for other l/m/x series models. This is because small models have limited capabilities and generally do not use strong data augmentation strategies like MixUp.

The core `Mosaic + RandomAffine + MixUp` process can be shown as follows:
(To be finished)

### 1.2 Network structure

This section was written by RangeKing@github. Thanks a lot!

The YOLOv5 network structure is the standard `CSPDarknet` + `PAFPN` + `non-decoupled Head`.

The size of the YOLOv5 network structure is determined by the `deepen_factor` and `widen_factor` parameters. `deepen_factor` controls the depth of the network structure, that is, the number of stacks of `DarknetBottleneck` modules in `CSPLayer`. `widen_factor` controls the width of the network structure, that is, the number of channels of the module output feature map. Take YOLOv5-l as an example. Its `deepen_factor = widen_factor = 1.0` , the overall structure is shown in the graph above.

The upper part of the figure is an overview of the model; the lower part is the specific network structure, in which the modules are marked with numbers in serial, which is convenient for users to correspond to the configuration files of the YOLOv5 official repository. The middle part is the detailed composition of each sub-module.

If you want to use \*\*netron \*\* to visualize the details of the network structure diagram, you can directly open the ONNX file format exported by MMDeploy in netron.

#### 1.2.1 Backbone

`CSPDarknet` in MMYOLO inherits from `BaseBackbone`. The overall structure is similar to `ResNet` with a total of 5 layers of structure, including one `Stem Layer` and four `Stage Layer`:

- `Stem Layer` is a `ConvModule` whose kernel size is 6x6. It is more efficient than the `Focus` module used before v6.1.
- Each of the first three `Stage Layer` consists of one `ConvModule` and one `CSPLayer`, as shown in the Details part in the graph above. `ConvModule` is a 3x3 `Conv2d` + `BatchNorm` + `SiLU activation function` module. `CSPLayer` is the C3 module in the official YOLOv5 repository, consisting of three `ConvModule` + n `DarknetBottleneck` with residual connections.
- The 4th `Stage Layer` adds an `SPPF` module at the end. The `SPPF` module is to serialize the input through multiple 5x5 `MaxPool2d` layers, which has the same effect as the `SPP` module but is faster.
- The P5 model passes the corresponding results from the second to the forth `Stage Layer` to the `Neck` structure and extracts three output feature maps. Take a 640x640 input image as an example. The output features are (B, 256, 80, 80), (B,512,40,40) and (B,1024,20,20), the corresponding stride is 8/16/32.

#### 1.2.2 Neck

There is no **Neck** part in the official YOLOv5. However, to facilitate users to correspond to other object detection networks easier, we split the `Head` of the official repository into `PAFPN` and `Head`.

Based on the `BaseYOLONeck` structure, YOLOv5's `Neck` also follows the same build process. For non-existed modules, we use `nn.Identity` instead.

The feature maps output by the Neck module are exactly the same as the Backbone, which are (B,256,80,80), (B,512,40,40) and (B,1024,20,20).

#### 1.2.3 Head

The `Head` structure of YOLOv5 is exactly the same as YOLOv3, which is a `non-decoupled Head`. The Head module includes three convolution modules that do not share weights. They are used only for input feature map transformation.

The `PAFPN` outputs three feature maps of different scales, whose shapes are (B,256,80,80), (B,512,40,40) and (B,1024,20,20) accordingly.

Since YOLOv5 has a non-decoupled output, that is, classification and bbox detection results are all in different channels of the same convolution module. Taking the COCO 80 class as an example, when the input is 640x640 resolution, the output shapes of the Head module are (B, 3x(4+1+80),80,80), (B, 3x(4+1+80),40,40) and (B, 3x(4+1+80),20,20). 3 represents three anchors, 4 represents the bbox prediction branch, 1 represents the obj prediction branch, and 80 represents the class prediction branch.

### 1.3 Positive and negative sample matching strategy

(To be finished)

### 1.4 Loss design

YOLOv5 contains a total of three Loss, which are:

- Classes loss: BCE loss
- Objectness loss: BCE loss
- Location loss: CIoU loss

These three losses are aggregated according to a certain proportion:

```{math}
Loss=\lambda_1L_{cls}+\lambda_2L_{obj}+\lambda_3L_{loc}
```

The Objectness loss corresponding to the P3, P4, and P5 layers are added according to different weights. The default setting is

```python
obj_level_weights=[4., 1., 0.4]
```

```{math}
L_{obj}=4.0\cdot L_{obj}^{small}+1.0\cdot L_{obj}^{medium}+0.4\cdot L_{obj}^{large}
```

In the reimplementation, we found a certain gap between the CIoU used in YOLOv5 and the latest official CIoU, which is reflected in the calculation of the alpha parameter.

In the official version:

Reference: https://github.com/Zzh-tju/CIoU/blob/master/layers/modules/multibox_loss.py#L53-L55

```python
alpha = (ious > 0.5).float() * v / (1 - ious + v)
```

In YOLOv5's version:

```python
alpha = v / (v - ious + (1 + eps))
```

This is an interesting detail, and we need to test the accuracy gap caused by different alpha calculation methods in our follow-up development.

### 1.5 Optimization and training strategies

YOLOv5 has very fine-grained control over the parameter groups of each optimizer, which briefly includes the following sections.

#### 1.5.1 Optimizer grouping

The optimization parameters are divided into three groups: Conv/Bias/BN. In the WarmUp stage, different groups use different lr and momentum update curves.
At the same time, the iter-based update strategy is adopted in the WarmUp stage, and it becomes an epoch-based update strategy in the non-WarmUp stage, which is quite tricky.

In MMYOLO, the YOLOv5OptimizerConstructor optimizer constructor is used to implement optimizer parameter grouping. The role of an optimizer constructor is to control the initialization process of some special parameter groups finely so that it can meet the needs well.

Different parameter groups use different scheduling curve functions through YOLOv5ParamSchedulerHook.

#### 1.5.2 weight decay parameter auto-adaptation

The author adopts different weight decay strategies for different batch sizes, specifically:

1. When the training batch size does not exceed 64, weight decay remains unchanged.
2. When the training batch size exceeds 64, weight decay will be linearly scaled according to the total batch size.

MMYOLO also implements through the YOLOv5OptimizerConstructor.

#### 1.5.3 Gradient accumulation

In order to maximize the performance under different batch sizes, the author sets the gradient accumulation function automatically when the total batch size is less than 64.

The training process is similar to most YOLO, including the following strategies:

1. Not using pre-trained weights.
2. There is no multi-scale training strategy, and cudnn.benchmark can be turned on to accelerate training further.
3. The EMA strategy is used to smooth the model.
4. Automatic mixed-precision training with AMP by default.

What needs to be reminded is that YOLOv5 officially uses single-card v100 training for the small model with a bs is 128. However, m/l/x models are trained with different numbers of multi-cards.
This training strategy is not relatively standard, **For this reason, eight cards are used in MMYOLO, and each card sets the bs to 16. At the same time, in order to avoid performance differences, SyncBN is turned on during training**.

### 1.6 Inference and post-processing

(To be finished)

## 2 Sum up

This article focuses on the principle of YOLOv5 and our implementation in MMYOLO in detail, hoping to help users understand the algorithm and the implementation process. At the same time, again, please note that since YOLOv5 itself is constantly being updated, this open-source library will also be constantly iterated. So please always check the latest version.
