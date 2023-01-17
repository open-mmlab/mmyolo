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

| Backbone | Arch | size | AMP | Mem (GB) | box AP |                                                  Config                                                   |        Download        |
| :------: | :--: | :--: | :-: | :------: | :----: | :-------------------------------------------------------------------------------------------------------: | :--------------------: |
| YOLOv8-n |  P5  | 640  | Yes |    x     |  37.2  | [config](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov8/yolov8_n_syncbn_8xb16-500e_coco.py) | [model](x) \| [log](x) |
| YOLOv8-s |  P5  | 640  | No  |    x     |  44.4  | [config](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov8/yolov8_s_syncbn_8xb16-500e_coco.py) | [model](x) \| [log](x) |
| YOLOv8-m |  P5  | 640  | Yes |    x     |  49.8  | [config](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov8/yolov8_m_syncbn_8xb16-500e_coco.py) | [model](x) \| [log](x) |
| YOLOv8-l |  P5  | 640  | Yes |    x     |   xx   | [config](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov8/yolov8_l_syncbn_8xb16-500e_coco.py) | [model](x) \| [log](x) |

**Note**
In the official YOLOv8 code, the [bbox annotation](https://github.com/ultralytics/ultralytics/blob/0cb87f7dd340a2611148fbf2a0af59b544bd7b1b/ultralytics/yolo/data/dataloaders/v5loader.py#L1011), [`random_perspective`](https://github.com/ultralytics/ultralytics/blob/0cb87f7dd3/ultralytics/yolo/data/dataloaders/v5augmentations.py#L208) and [`copy_paste`](https://github.com/ultralytics/ultralytics/blob/0cb87f7dd3/ultralytics/yolo/data/dataloaders/v5augmentations.py#L208) data augmentation in COCO object detection task training uses mask annotation information, which leads to higher performance. Object detection should not use mask annotation, so only box annotation information is used in `MMYOLO`. In order to align with the official performance, we will support this feature in the next version.

1. We use 8x A100 for training, and the single-GPU batch size is 16. This is different from the official code, but has no effect on performance.
2. The performance is unstable and may fluctuate by about 0.3 mAP and the highest performance weight in `COCO` training in `YOLOv8` may not be the last epoch.
3. We provide a script for [official weight transfer to MMYOLO](https://github.com/open-mmlab/mmyolo/tree/dev/tools/model_converters/yolov8_to_mmyolo.py)
4. `SyncBN` means use SyncBN, `AMP` indicates training with mixed precision.

## Citation
