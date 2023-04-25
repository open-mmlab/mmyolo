# Algorithm principles and implementation with YOLOv8

## 0 Introduction

<div align=center >
<img alt="YOLOv8-P5_structure" src="https://user-images.githubusercontent.com/27466624/222869864-1955f054-aa6d-4a80-aed3-92f30af28849.jpg"/>
Figure 1：YOLOv8-P5
</div>

RangeKing@github provides the graph above. Thanks, RangeKing!

YOLOv8 is the next major update from YOLOv5, open sourced by Ultralytics on 2023.1.10, and now supports image classification, object detection and instance segmentation tasks.

<div align=center >
<img alt="YOLOv8-logo" src="https://user-images.githubusercontent.com/17425982/212823787-44031e62-e374-4851-8267-4e56e299473a.png"/>
Figure 2：YOLOv8-logo
</div>
According to the official description, Ultralytics YOLOv8 is the latest version of the YOLO object detection and image segmentation model developed by Ultralytics. YOLOv8 is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. These include a new backbone network, a new anchor-free detection head, and a new loss function. YOLOv8 is also highly efficient and can be run on a variety of hardware platforms, from CPUs to GPUs.

However, instead of naming the open source library YOLOv8, ultralytics uses the word ultralytics directly because ultralytics positions the library as an algorithmic framework rather than a specific algorithm, with a major focus on scalability. It is expected that the library can be used not only for the YOLO model family, but also for non-YOLO models and various tasks such as classification segmentation pose estimation.

Overall, YOLOv8 is a powerful and flexible tool for object detection and image segmentation that offers the best of both worlds: **the SOTA technology and the ability to use and compare all previous YOLO versions.**

<div align=center >
<img alt="YOLOv8-table" src="https://user-images.githubusercontent.com/17425982/212007736-f592bc70-3959-4ff6-baf7-a93c7ad1d882.png"/>
Figure 3：YOLOv8-performance
</div>

YOLOv8 official open source address: [this](https://github.com/ultralytics/ultralytics)

MMYOLO open source address for YOLOv8: [this](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov8/)

The following table shows the official results of mAP, number of parameters and FLOPs tested on the COCO Val 2017 dataset. It is evident that YOLOv8 has significantly improved precision compared to YOLOv5. However, the number of parameters and FLOPs of the N/S/M models have significantly increased. Additionally, it can be observed that the inference speed of YOLOv8 is slower in comparison to most of the YOLOv5 models.

| **model** | **YOLOv5**  | **params(M)** | **FLOPs@640 (B)** | **YOLOv8**  | **params(M)** | **FLOPs@640 (B)** |
| --------- | ----------- | ------------- | ----------------- | ----------- | ------------- | ----------------- |
| n         | 28.0(300e)  | 1.9           | 4.5               | 37.3 (500e) | 3.2           | 8.7               |
| s         | 37.4 (300e) | 7.2           | 16.5              | 44.9 (500e) | 11.2          | 28.6              |
| m         | 45.4 (300e) | 21.2          | 49.0              | 50.2 (500e) | 25.9          | 78.9              |
| l         | 49.0 (300e) | 46.5          | 109.1             | 52.9 (500e) | 43.7          | 165.2             |
| x         | 50.7 (300e) | 86.7          | 205.7             | 53.9 (500e) | 68.2          | 257.8             |

It is worth mentioning that the recent YOLO series have shown significant performance improvements on the COCO dataset. However, their generalizability on custom datasets has not been extensively tested, which thereby will be a focus in the future development of MMYOLO.

Before reading this article, if you are not familiar with YOLOv5, YOLOv6 and RTMDet, you can read the detailed explanation of [YOLOv5 and its implementation](https://mmyolo.readthedocs.io/en/latest/algorithm_descriptions/yolov5_description.html).

## 1 YOLOv8 Overview

The core features and modifications of YOLOv8 can be summarized as follows:

1. **A new state-of-the-art (SOTA) model is proposed, featuring an object detection model for P5 640 and P6 1280 resolutions, as well as a YOLACT-based instance segmentation model. The model also includes different size options with N/S/M/L/X scales, similar to YOLOv5, to cater to various scenarios.**
2. **The backbone network and neck module are based on the YOLOv7 ELAN design concept, replacing the C3 module of YOLOv5 with the C2f module. However, there are a lot of operations such as Split and Concat in this C2f module that are not as deployment-friendly as before.**
3. **The Head module has been updated to the current mainstream decoupled structure, separating the classification and detection heads, and switching from Anchor-Based to Anchor-Free.**
4. **The loss calculation adopts the TaskAlignedAssigner in TOOD and introduces the Distribution Focal Loss to the regression loss.**
5. **In the data augmentation part, Mosaic is closed in the last 10 training epoch, which is the same as YOLOX training part.**
   **As can be seen from the above summaries, YOLOv8 mainly refers to the design of recently proposed algorithms such as YOLOX, YOLOv6, YOLOv7 and PPYOLOE.**

Next, we will introduce various improvements in the YOLOv8 model in detail by 5 parts: model structure design, loss calculation, training strategy, model inference process and data augmentation.

## 2 Model structure design

The Figure 1 is the model structure diagram based on the official code of YOLOv8. **If you like this style of model structure diagram, welcome to check out the model structure diagram in algorithm README of MMYOLO, which currently covers YOLOv5, YOLOv6, YOLOX, RTMDet and YOLOv8.**

Comparing the YOLOv5 and YOLOv8 yaml configuration files without considering the head module, you can see that the changes are minor.

<div align=center >
<img alt="yaml" src="https://user-images.githubusercontent.com/17425982/212008977-28c3fc7b-ee00-4d56-b912-d77ded585d78.png"/>
Figure 4：YOLOv5 and YOLOv8 YAML diff
</div>

The structure on the left is YOLOv5-s and the other side is YOLOv8-s. The specific changes in the backbone network and neck module are:

- The kernel of the first convolutional layer has been changed from 6x6 to 3x3
- All C3 modules are replaced by C2f, and the structure is as follows, with more skip connections and additional split operations.

<div align=center >
<img alt="module" src="https://user-images.githubusercontent.com/17425982/212009208-92f45c23-a024-49bb-a2ee-bb6f87adcc92.png"/>
Figure 5：YOLOv5 and YOLOv8 module diff
</div>

- Removed 2 convolutional connection layers from neck module
- The block number has been changed from 3-6-9-3 to 3-6-6-3.
- **If we look at the N/S/M/L/X models, we can see that of the N/S and L/X models only changed the scaling factors, but the number of channels in the S/ML backbone network is not the same and does not follow the same scaling factor principle. The main reason for this design is that the channel settings under the same set of scaling factors are not the most optimal, and the YOLOv7 network design does not follow one set of scaling factors for all models either.**

The most significant changes in the model lay in the head module. The head module has been changed from the original coupling structure to the decoupling one, and its style has been changed from **YOLOv5's Anchor-Based to Anchor-Free**. The structure is shown below.

<div align=center >
<img alt="head" src="https://user-images.githubusercontent.com/17425982/212009547-189e14aa-6f93-4af0-8446-adf604a46b95.png"/>
Figure 6：YOLOv8 Head
</div>

As demonstrated, the removal of the objectness branch and the retention of only the decoupled classification and regression branches stand as the major differences. Additionally, the regression branch now employs integral form representation as proposed in the Distribution Focal Loss.

## 3 Loss calculation

The loss calculation process consists of 2 parts: the sample assignment strategy and loss calculation.

The majority of contemporary detectors employ dynamic sample assignment strategies, such as YOLOX's simOTA, TOOD's TaskAlignedAssigner, and RTMDet's DynamicSoftLabelAssigner. Given the superiority of dynamic assignment strategies, the YOLOv8 algorithm directly incorporates the one employed in TOOD's TaskAlignedAssigner.

The matching strategy of TaskAlignedAssigner can be summarized as follows: positive samples are selected based on the weighted scores of classification and regression.

```{math}
t=s^\alpha+u^\beta
```

`s` is the prediction score corresponding to the ground truth category, `u` is the IoU of the prediction bounding box and the gt bounding box.

1. For each ground truth, the task-aligned assigner calculates the `alignment metric` for each anchor by taking the weighted product of two values: the predicted classification score of the corresponding class, and the Intersection over Union (IoU) between the predicted bounding box and the Ground Truth bounding box.
2. For each Ground Truth, the larger top-k samples are selected as positive based on the `alignment_metrics` values directly.

The loss calculation consists of 2 parts: the classification and regression, without the objectness loss in the previous model.

- The classification branch still uses BCE Loss.
- The regression branch employs both Distribution Focal Loss and CIoU Loss.

The 3 Losses are weighted by a specific weight ratio.

## 4 Data augmentation

YOLOv8's data augmentation is similar to YOLOv5, whereas it stops the Mosaic augmentation in the final 10 epochs as proposed in YOLOX. The data process pipelines are illustrated in the diagram below.

<div align=center >
<img alt="head" src="https://user-images.githubusercontent.com/17425982/212815248-38384da9-b289-468e-8414-ab3c27ee2026.png"/>
Figure 7：pipeline
</div>

The intensity of data augmentation required for different scale models varies, therefore the hyperparameters for the scaled models are adjusted depending on the situation. For larger models, techniques such as MixUp and CopyPaste are typically employed. The result of data augmentation can be seen in the example below:

<div align=center >
<img alt="head" src="https://user-images.githubusercontent.com/17425982/212815840-063524e1-d754-46b1-9efc-61d17c03fd0e.png"/>
Figure 8：results
</div>

The above visualization result can be obtained by running the [browse_dataset](https://github.com/open-mmlab/mmyolo/blob/dev/tools/analysis_tools/browse_dataset.py) script.

As the data augmentation process utilized in YOLOv8 is similar to YOLOv5, we will not delve into the specifics within this article. For a more in-depth understanding of each data transformation, we recommend reviewing the [YOLOv5 algorithm analysis document](https://mmyolo.readthedocs.io/en/latest/algorithm_descriptions/yolov5_description.html#id2) in MMYOLO.

## 5 Training strategy

The distinctions between the training strategy of YOLOv8 and YOLOv5 are minimal. The most notable variation is that the overall number of training epochs for YOLOv8 has been raised from 300 to 500, resulting in a significant expansion in the duration of training. As an illustration, the training strategy for YOLOv8-S can be succinctly outlined as follows:

| config                 | YOLOv8-s P5 hyp                 |
| ---------------------- | ------------------------------- |
| optimizer              | SGD                             |
| base learning rate     | 0.01                            |
| Base weight decay      | 0.0005                          |
| optimizer momentum     | 0.937                           |
| batch size             | 128                             |
| learning rate schedule | linear                          |
| training epochs        | **500**                         |
| warmup iterations      | max(1000，3 * iters_per_epochs) |
| input size             | 640x640                         |
| EMA decay              | 0.9999                          |

## 6 Inference process

The inference process of YOLOv8 is almost the same as YOLOv5. The only difference is that the integral representation bbox in Distribution Focal Loss needs to be decoded into a regular 4-dimensional bbox, and the subsequent calculation process is the same as YOLOv5.

Taking COCO 80 class as an example, assuming that the input image size is 640x640, the inference process implemented in MMYOLO is shown as follows.

<div align=center >
<img alt="head" src="https://user-images.githubusercontent.com/17425982/212816206-33815716-3c12-49a2-9c37-0bd85f941bec.png"/>
Figure 9：results
</div>
The inference and post-processing process is:

**(1) Decoding bounding box**
Integrate the probability of the distance between the center and the boundary of the box into the mathematical expectation of the distances.

**(2) Dimensional transformation**
YOLOv8 outputs three feature maps with `80x80`, `40x40` and `20x20` scales. A total of 6 classification and regression different scales of  feature map are output by the head module.
The 3 different scales of category prediction branch and bbox prediction branch are combined and dimensionally transformed. For the convenience of subsequent processing, the original channel dimensions are transposed to the end, and the category prediction branch and bbox prediction branch shapes are (b, 80x80+40x40+20x20, 80)=(b,8400,80), (b,8400,4), respectively.

**(3) Scale Restroation**
The classification prediction branch utilizes sigmoid calculations, whereas the bbox prediction branch requires decoding to xyxy format and conversion to the original scale of the input images.

**(4) Thresholding**
Iterate through each graph in the batch and use `score_thr` to perform thresholding. In this process, we also need to consider multi_label and nms_pre to ensure that the number of detected bboxs after filtering is no more than nms_pre.

**(5) Reduction to the original image scale and NMS**
Reusing the parameters for preprocessing, the remaining bboxs are first resized to the original image scale and then NMS is performed. The final number of bboxes cannot be more than `max_per_img`.

Special Note: **The Batch shape inference strategy, which is present in YOLOv5, is currently not activated in YOLOv8. By performing a quick test in MMYOLO, it can be observed that activating the Batch shape strategy can result in an approximate AP increase of around 0.1% to 0.2%.**

## 7 Feature map visualization

A comprehensive set of feature map visualization tools are provided in MMYOLO to help users visualize the feature maps.

Take the YOLOv8-s model as an example. The first step is to download the official weights, and then convert them to MMYOLO by using the [yolov8_to_mmyolo](https://github.com/open-mmlab/mmyolo/blob/dev/tools/model_converters/yolov8_to_mmyolo.py) script. Note that the script must be placed under the official repository in order to run correctly.

Assuming that you want to visualize the effect of the 3 feature maps output by backbone and the weights are named 'mmyolov8s.pth'. Run the following command:

```bash
cd mmyolo
python demo/featmap_vis_demo.py demo/demo.jpg configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py mmyolov8s.pth --channel-reductio squeeze_mean
```

In particular, to ensure that the feature map and image are shown aligned, the original `test_pipeline` configuration needs to be replaced with the following:

```Python
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False), # change
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
```

<div align=center >
<img alt="head" src="https://user-images.githubusercontent.com/17425982/212816319-9ac19484-987a-40ac-a0fe-2c13a7048df7.png"/>
Figure 10：featmap
</div>
From the above figure, we can see that the different output feature maps are mainly responsible for predicting objects at different scales.
We can also visualize the 3 output feature maps of the neck layer.

```bash
cd mmyolo
python demo/featmap_vis_demo.py demo/demo.jpg configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py mmyolov8s.pth --channel-reductio squeeze_mean --target-layers neck
```

<div align=center >
<img alt="head" src="https://user-images.githubusercontent.com/17425982/212816458-a4e4600a-5f50-49c6-864b-0254a2720f3c.png"/>
Figure 11：featmap
</div>

From the above figure, we can find the features at the object are more focused.

## Summary

This article delves into the intricacies of the YOLOv8 algorithm, offering a comprehensive examination of its overall design, model structure, loss function, training data enhancement techniques, and inference process. To aid in comprehension, a plethora of diagrams are provided.

In summary, YOLOv8 is a highly efficient algorithm that incorporates image classification, Anchor-Free object detection, and instance segmentation. Its detection component incorporates numerous state-of-the-art YOLO algorithms to achieve new levels of performance.

MMYOLO open source address for YOLOV8 [this](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov8/)

MMYOLO Algorithm Analysis Tutorial address is [yolov5_description](https://mmyolo.readthedocs.io/en/latest/algorithm_descriptions/yolov5_description.html)
