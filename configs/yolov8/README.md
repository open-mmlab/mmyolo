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

| Backbone | Arch | size | Mask Refine | SyncBN | AMP | Mem (GB) |   box AP    | TTA box AP |                                 Config                                  |                                                                                                                                                                                   Download                                                                                                                                                                                   |
| :------: | :--: | :--: | :---------: | :----: | :-: | :------: | :---------: | :--------: | :---------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOv8-n |  P5  | 640  |     No      |  Yes   | Yes |   2.8    |    37.2     |            |       [config](../yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804-88c11cdb.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco/yolov8_n_syncbn_fast_8xb16-500e_coco_20230114_131804.log.json)                         |
| YOLOv8-n |  P5  | 640  |     Yes     |  Yes   | Yes |   2.5    | 37.4 (+0.2) |    39.9    | [config](../yolov8/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_101206-b975b1cd.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_101206.log.json) |
| YOLOv8-s |  P5  | 640  |     No      |  Yes   | Yes |   4.0    |    44.2     |            |       [config](../yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101.log.json)                         |
| YOLOv8-s |  P5  | 640  |     Yes     |  Yes   | Yes |   4.0    | 45.1 (+0.9) |    46.8    | [config](../yolov8/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_095938-ce3c1b3f.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_095938.log.json) |
| YOLOv8-m |  P5  | 640  |     No      |  Yes   | Yes |   7.2    |    49.8     |            |       [config](../yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200.log.json)                         |
| YOLOv8-m |  P5  | 640  |     Yes     |  Yes   | Yes |   7.0    | 50.6 (+0.8) |    52.3    | [config](../yolov8/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_223400-f40abfcd.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_223400.log.json) |
| YOLOv8-l |  P5  | 640  |     No      |  Yes   | Yes |   9.8    |    52.1     |            |       [config](../yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526.log.json)                         |
| YOLOv8-l |  P5  | 640  |     Yes     |  Yes   | Yes |   9.1    | 53.0 (+0.9) |    54.4    | [config](../yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120100-5881dec4.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120100.log.json) |
| YOLOv8-x |  P5  | 640  |     No      |  Yes   | Yes |   12.2   |    52.7     |            |       [config](../yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338.log.json)                         |
| YOLOv8-x |  P5  | 640  |     Yes     |  Yes   | Yes |   12.4   | 54.0 (+1.3) |    55.0    | [config](../yolov8/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120411-079ca8d1.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120411.log.json) |

**Note**

1. We use 8x A100 for training, and the single-GPU batch size is 16. This is different from the official code, but has no effect on performance.
2. The performance is unstable and may fluctuate by about 0.3 mAP and the highest performance weight in `COCO` training in `YOLOv8` may not be the last epoch. The performance shown above is the best model.
3. We provide [scripts](https://github.com/open-mmlab/mmyolo/tree/dev/tools/model_converters/yolov8_to_mmyolo.py) to convert official weights to MMYOLO.
4. `SyncBN` means using SyncBN, `AMP` indicates training with mixed precision.
5. The performance of `Mask Refine` training is for the weight performance officially released by YOLOv8. `Mask Refine` means refining bbox by mask while loading annotations and transforming after `YOLOv5RandomAffine`, and the L and X models use `Copy Paste`.
6. `TTA` means that Test Time Augmentation. It's perform 3 multi-scaling transformations on the image, followed by 2 flipping transformations (flipping and not flipping). You only need to specify `--tta` when testing to enable.  see [TTA](https://github.com/open-mmlab/mmyolo/blob/dev/docs/en/common_usage/tta.md) for details.

## Citation
