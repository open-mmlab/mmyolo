# YOLOv8

<!-- [ALGORITHM] -->

## Abstract

Ultralytics YOLOv8, developed by Ultralytics, is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection, image segmentation and image classification tasks.

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/212812246-51dc029c-e892-455d-86b4-946b5d03957a.png"/>
performance
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/211974251-8de633c8-090c-47c9-ba52-4941dc9e3a48.jpg"/>
YOLOv8-P5 model structure
</div>

## Results and models

### COCO

| Backbone | Arch | size | SyncBN | AMP | Mem (GB) | box AP |                                                     Config                                                     |                                                                                                                                                           Download                                                                                                                                                           |
| :------: | :--: | :--: | :----: | :-: | :------: | :----: | :------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOv8-n |  P5  | 640  |  Yes   | Yes |   2.8    |  37.2  | [config](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804.log.json) |
| YOLOv8-s |  P5  | 640  |  Yes   | Yes |   4.0    |  44.2  | [config](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101.log.json) |
| YOLOv8-m |  P5  | 640  |  Yes   | Yes |   7.2    |  49.8  | [config](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200.log.json) |

**Note**

In the official YOLOv8 code, the [bbox annotation](https://github.com/ultralytics/ultralytics/blob/0cb87f7dd340a2611148fbf2a0af59b544bd7b1b/ultralytics/yolo/data/dataloaders/v5loader.py#L1011), [`random_perspective`](https://github.com/ultralytics/ultralytics/blob/0cb87f7dd3/ultralytics/yolo/data/dataloaders/v5augmentations.py#L208) and [`copy_paste`](https://github.com/ultralytics/ultralytics/blob/0cb87f7dd3/ultralytics/yolo/data/dataloaders/v5augmentations.py#L208) data augmentation in COCO object detection task training uses mask annotation information, which leads to higher performance. Object detection should not use mask annotation, so only box annotation information is used in `MMYOLO`. We trained the official YOLOv8s code with `8xb16` configuration and its best performance is also 44.2. We will support mask annotations in object detection tasks in the next version.

1. We use 8x A100 for training, and the single-GPU batch size is 16. This is different from the official code, but has no effect on performance.
2. The performance is unstable and may fluctuate by about 0.3 mAP and the highest performance weight in `COCO` training in `YOLOv8` may not be the last epoch. The performance shown above is the best model.
3. We provide [scripts](https://github.com/open-mmlab/mmyolo/tree/dev/tools/model_converters/yolov8_to_mmyolo.py) to convert official weights to MMYOLO.
4. `SyncBN` means use SyncBN, `AMP` indicates training with mixed precision.

## Citation
