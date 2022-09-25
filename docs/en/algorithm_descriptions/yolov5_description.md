# YOLOv5 principle and implementation

## 0 Introduction

<div align=center >
<img alt="YOLOv5_structure_v3" src="https://user-images.githubusercontent.com/27466624/190986656-b68bdb59-d0d5-480c-83c2-44f5648320a2.jpg"/>
</div>

The above structure diagram is drawn by **RangeKing@github**.

YOLOv5 is an open-source object detection algorithm for real-time industrial applications which has received extensive attention. We believe that the reason for the explosion of YOLOv5 is not simply the excellence of the YOLOv5 algorithm itself. It's more about the utility and robustness of the open-source library.

In short, the main features of the YOLOv5 open-source library are:

1. **Friendly and perfect deployment support**
2. **The algorithm training speed is extremely fast**, the training time in the case of 300 epoch and most one-stage algorithms such as RetinaNet, ATSS and two-stage algorithms such as Faster R-CNN 12 epoch time approaching
3. The framework has carried out **a lot of corner case optimization**, and features and documents are also richer

This article will start with the principle of the YOLOv5 algorithm itself and then focus on analyzing the implementation in MMYOLO. Please read the follow-up documents for the usage guide and speed comparison of YOLOv5.

We hope this article becomes the core document for you to start and master YOLOv5. As YOLOv5 itself is being constantly and iteratively updated, we will also keep updating this document. Please always check the latest version.

MMYOLO implementation configuration: https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/

YOLOv5 official open-source library address: https://github.com/ultralytics/yolov5

## 1 v6.1 algorithm principle and MMYOLO implementation analysis

YOLOv5 official release address: https://github.com/ultralytics/yolov5/releases/tag/v6.1

<div align=center >
<img alt="YOLOv5 accuracy map" src="https://user-images.githubusercontent.com/40284075/190542120-29d46b7e-ce3c-436a-9933-cfc9f86787bf.png"/>
</div>

<div align=center >
<img alt="YOLOv5 Accuracy Speed Map" src="https://user-images.githubusercontent.com/40284075/190542279-37734629-2b59-4bd8-a9bf-757875a93eed.png"/>
</div>

The performance is shown in the table above. YOLOv5 has two models with different training input scales, P5 and P6. P6 is a larger model with 1280x1280 input. Usually, the P5 will be used, and the input size is 640x640. This article also interprets the structure of the P5 model.

The target detection algorithms can usually be divided into the following components: data enhancement, model structure, loss calculation, etc. It is the same for YOLOv5, as shown below:

<div align=center >
<img alt="Train Test Strategy" src="https://user-images.githubusercontent.com/40284075/190542423-f6b20d8e-c82a-4a34-9065-c161c5e29e7c.png"/>
</div>

The following part will briefly analyze the principle of YoloV5 with the specific implementation of MMYOLO.

### 1.1 Data Enhancement Module

Many data augmentation methods are used in the YOLOv5, including:

- **Mosaic**
- **RandomAffine**
- **MixUp**
- **Image blur and other transformations implemented with Albu**
- **HSV color space enhancements**
- **Random horizontal flips**

Among them, the Mosaic data enhancement probability is `1`, which will always be triggered. MixUp is not used for the small and nano models, and the rest series (l/m/x) set the probability to `0.1` to trigger the MixUp. Small models have limited capabilities and usually will not use strong data enhancement strategies like MixUp.

Its core `Mosaic + RandomAffine + MixUp` process can be drawn briefly as follows:

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190542598-bbf4a159-cc9d-4bac-892c-46ef99267994.png"/>
</div>

#### 1.1.1 Mosaic

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190542619-d777894f-8928-4244-b39a-158eea416ccd.png"/>
</div>

Mosaic is one kind of mixed data augmentation because it requires 4 image stitching at the runtime, equivalent to increasing the training batch size in disguise. A brief overview of its operation process is as follows:

1. Randomly generates the coordinates of the junction center point for the 4 images after splicing, which is equivalent to determining the junction point of the 4 spliced images
2. Randomly retrieves the other 3 images and read the corresponding annotations
3. Resizes each image to the specified setting according to its aspect ratio
4. Calculates the position where each picture should be placed in the output image based on the up, down, left, and right rules. As the image may be out of bounds, it will be necessary to calculate the crop coordinates.
5. Uses the crop coordinates to crop the scaled image and places it back to where we previously calculated. Then fill the rest of the position with the value of 114-pixel.
6. The annotation of each image will be processed accordingly

Note: Since 4 images are spliced, the output area will be expanded by 4 times, from 640x640 to 1280x1280. To restore to 640x640, there must be another **RandomAffine** transformation. Otherwise, the image area will always be 4 times larger.

#### 1.1.2 RandomAffine

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190542871-14e91a42-329f-4084-aec5-b3e412e5364b.png"/>
</div>

RandomAffine transformation serves two primary purposes:

1. Perform random geometric affine transformation on the image
2. Restore the `4x` enlarged image output by Mosaic to 640x640 size

RandomAffine includes geometric enhancement operations such as translation, rotation, scaling, and staggering. Moreover, Mosaic and RandomAffine are relatively strong enhancement operations that will introduce considerable noise. Therefore, the enhanced annotation needs to be processed, and the filtering rule is:

1. The height and width of the enhanced `gt bbox` should be larger than `wh_thr`.
2. The area of `gt bbox` before enhancement and the area of `gt bbox` after enhancement should be larger than `ar_thr` to prevent too serious enhancement.
3. The maximum aspect ratio should be smaller than `area_thr` to prevent the aspect ratio from changing too much.

Since the annotation box will become larger after rotation, leading to inaccuracy, rotation data enhancement is rarely used in object detection.

#### 1.1.3 MixUp

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190543076-db60e4b2-0552-4cf4-ab45-259d1ccbd5a6.png"/>
</div>

Like Mosaic, MixUp is also a mixed image enhancement. It first randomly selects an image. Then the two images are randomly mixed. There are many ways to achieve it. The common practice is:
Either the labels are spliced directly, or the labels also use alpha blending. The author's approach is very simple. The labels are spliced directly, and the images are mixed by distribution sampling.

Note:
**In the MixUp of YOLOv5, the other image randomly selected must be enhanced by `Mosaic + RandomAffine` before the mixing as well. This is not the same as some implementations from other open-sourced libraries**.

### 1.1.4 Image blur and other data enhancements

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190543533-8b9ece51-676b-4a7d-a7d0-597e2dd1d42e.png"/>
</div>

The remaining data enhancements include:

- **Image blur and other transformations implemented with Albu**
- **HSV color space enhancements**
- **Random horizontal flips**

The `Albu` third-party data augmentation library has been encapsulated in the MMDetection open-sourced library so that users can use any data augmentation functions provided in the `Albu` library simply through configuration.
As HSV color space enhancement and random horizontal flip are relatively conventional data enhancements, there will be no special introduction.

#### 1.1.5 MMYOLO implementation analysis

Conventional single-image data augmentation, such as random flipping, is relatively easy to implement, while Mosaic-like mixed data augmentation is not so easy. In the YOLOX algorithm reproduced by MMDetection, the concept of `MultiImageMixDataset` dataset wrapper is proposed, and its implementation process is as follows:

<div align=center >
<img alt="image" src="https://user-images.githubusercontent.com/40284075/190543666-d5a22ed7-46a0-4696-990a-12ebde7f8907.png"/>
</div>

For mixed data enhancement such as Mosaic, an additional `get_indexes` method is proposed to obtain indexes of other images. Then Mosaic enhancement can be performed after obtaining 4 image information.
Taking YOLOX implemented in MMDetection as an example, its configuration file is written as follows:

```python
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    ...
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    pipeline=train_pipeline)
```
