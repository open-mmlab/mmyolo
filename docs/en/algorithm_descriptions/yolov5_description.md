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
