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
<img alt="Strategy" src="https://user-images.githubusercontent.com/40284075/190542423-f6b20d8e-c82a-4a34-9065-c161c5e29e7c.png"/>
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

If you want to use **netron** to visualize the details of the network structure, just open the ONNX file format exported by MMDeploy in netron.

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

Since YOLOv5 has a non-decoupled output, that is, classification and bbox detection results are all in different channels of the same convolution module. Taking the COCO dataset as an example, when the input is 640x640 resolution, the output shapes of the Head module are `(B, 3x(4+1+80),80,80)`, `(B, 3x(4+1+80),40,40)` and `(B, 3x(4+1+80),20,20)`. `3` represents three anchors, `4` represents the bbox prediction branch, `1` represents the obj prediction branch, and `80` represents the class prediction branch of the COCO dataset.

### 1.3 Positive and negative sample matching strategy

The core of the positive and negative sample matching strategy is to determine which positions in all positions of the predicted feature map should be positive or negative and even which samples will be ignored.

This is the one of the most significant components of the object detection algorithm because a good strategy can improve the algorithm's performance.

The matching strategy of YOLOv5 can be briefly summarized as calculating the shape-matching rate between anchor and gt_bbox. Plus, the cross-neighborhood grid is also introduced to get more positive samples.

It consists the following two main steps:

1. For any output layer, instead of the commonly used strategy based on Max IoU matching, YOLOv5 switched to comparing the shape matching ratio. First, the GT Bbox and the anchor of the current layer are used to calculate the aspect ratio. If the ratio is greater than the threshold, the GT Bbox and Anchor are considered not matched. Then the current GT Bbox is temporarily discarded, and the predicted position in the grid of this GT Bbox in the current layer is considered as a negative sample.
2. For the remaining GT Bboxes (the matched GT Bboxes) YOLOv5 calculates which grid they fall in. Using the rounding rule to find the nearest two grids and considering all three grids as a group which is responsible for predicting the GT Bbox. We can be roughly estimate that the number of positive samples has increased by at least three times compared to the previous YOLO series algorithms.

Now we will explain each part of the matching strategy in detail. Some descriptions and illustrations are either directly or indirectly referenced from the official [repo](https://github.com/ultralytics/YOLOv5/issues/6998#44).

#### 1.3.1 Anchor settings

YOLOv5 is an anchor-based object detection algorithm. The anchor sizes are obtained in the same way as YOLOv3, which is by clustering using the K-means algorithm.

When users change the data, they can use the anchor analysis tool in MMDetection to determine the appropriate anchor size for their dataset.

If MMDetection is installed via `mim`, you can use the following command to analyze the anchor:

```shell
mim run mmdet optimize_anchors ${CONFIG} --algorithm k-means
--input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} --output-dir ${OUTPUT_DIR}
```

If MMDetection is installed in other ways, you can go to MMDetection directory and use the following command to analyze the anchor:

```shell
python tools/analysis_tools/optimize_anchors.py ${CONFIG} --algorithm k-means
  --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} --output-dir ${OUTPUT_DIR}
```

Then modify the default anchor size setting in the [config file](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py):

```python
anchors = [[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45), (59, 119)],
           [(116, 90), (156, 198), (373, 326)]]
```

#### 1.3.2 Bbox encoding and decoding process

In anchor-based algorithms, the predicted bounding box will transform based on the pre-set anchors. Then, the transformation amount is predicted, known as the GT Bbox encoding process. Finally, the Pred Bbox decoding needs to be performed after the prediction to restore the bboxes to the original scale, known as the Pred Bbox decoding process.

In YOLOv3, the bbox regression formula is:

```{math}
b_x=\sigma(t_x)+c_x  \\
b_y=\sigma(t_y)+c_y  \\
b_w=a_w\cdot e^{t_w} \\
b_h=a_h\cdot e^{t_h} \\
```

In the above formula，

```{math}
a_w represents the width of the anchor \\
c_x represents the coordinate of the grid \\
\sigma represents the Sigmoid function.
```

However, the regression formula in YOLOv5 is：

```{math}
b_x=(2\cdot\sigma(t_x)-0.5)+c_x   \\
b_y=(2\cdot\sigma(t_y)-0.5)+c_y   \\
b_w=a_w\cdot(2\cdot\sigma(t_w))^2   \\
b_h=a_h\cdot(2\cdot\sigma(t_h))^2
```

Two main changes are:

- adjusted the range of the center point coordinate from (0, 1) to (-0.5, 1.5);
- adjusted the width and height from

```{math}
(0，+\infty)
```

to

```{math}
(0，4a_{wh})
```

The changes have the two benefits:

- It will be **better to predict zero and one** with the changed center point range, which makes the bbox coordinate regression more accurate.

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190546778-83001bac-4e71-4b9a-8de8-bd41146495af.png"/>
</div>

- `exp(x)` in the width and height regression formula is unbounded, which may cause the **gradient out of control** and make the training stage unstable. The revised width-height regression in YOLOv5 optimizes this problem.

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190546793-5364d6d3-7891-4af3-98e3-9f06970f3163.png"/>
</div>

#### 1.3.3 Matching strategy

Note: in MMYOLO, **we call anchor as prior** for both anchor-based and anchor-free networks.

Positive sample matching consists of the following two steps:

(1) Scale comparison

Compare the scale of the WH in the GT BBox and the WH in the Prior:

```{math}
r_w = w\_{gt} / w\_{pt}    \\
r_h = h\_{gt} / h\_{pt}    \\
r_w^{max}=max(r_w, 1/r_w)  \\
r_h^{max}=max(r_h, 1/r_h)  \\
r^{max}=max(r_w^{max}, r_h^{max})   \\
if\ \ r_{max} < prior\_match\_thr:   match!
```

Taking the matching process of the GT Bbox and the Prior of the P3 feature map as the example:

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190547195-60d6cd7a-b12a-4c6f-9cc8-13f48c8ab1e0.png"/>
</div>

The reason why Prior 1 fails to match the GT Bbox is because:

```{math}
h\_{gt}\ /\ h\_{prior}\ =\ 4.8\ >\ prior\_match\_thr
```

(2) Assign corresponded positive samples to the matched GT BBox in step 1

We still use the example in the previous step.

The value of (cx, cy, w, h) of the GT BBox is (26, 37, 36, 24), and the WH value of the Prior is \[(15, 5), (24, 16), (16, 24)\]. In the P3 feature map, the stride is eight. Prior 2 and prior 3 are matched.

The detailed process can be described as:

(2.1) Map the center point coordinates of the GT Bbox to the grid of P3.

```{math}
GT_x^{center_grid}=26/8=3.25  \\
GT_y^{center_grid}=37/8=4.625
```

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190549304-020ec19e-6d54-4d40-8f43-f78b8d6948aa.png"/>
</div>

(2.2) Divide the grid where the center point of GT Bbox locates into four quadrants. **Since the center point falls in the lower left quadrant, the left and lower grids of the object will also be considered positive samples**.

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190549310-e5da53e3-eae3-4085-bd0a-1843ac8ca653.png"/>
</div>

The following picture shows the distribution of positive samples when the center point falls to different positions:

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190549613-eb47e70a-a2c1-4729-9fb7-f5ce7007842b.png"/>
</div>

So what improvements does the Assign method bring to YOLOv5?

- One GT Bbox can match multiple Priors.

- When a GT Bbox matches a Prior, at most three positive samples can be assigned.

- These strategies can **moderately alleviate the problem of unbalanced positive and negative samples, which is very common in object detection algorithms**.

The regression method in YOLOv5 corresponds to the Assign method:

1. Center point regression:

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190549684-21776c33-9ef8-4818-9530-14f750a18d63.png"/>
</div>

2. WH regression:

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190549696-3da08c06-753a-4108-be47-64495ea480f2.png"/>
</div>

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
