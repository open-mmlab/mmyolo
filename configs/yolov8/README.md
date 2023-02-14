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

| Backbone | Arch | size | Mask Refine | Copy Paste | SyncBN | AMP | Mem (GB) | box AP |                                 Config                                  |        Download        |
| :------: | :--: | :--: | :---------: | :--------: | :----: | :-: | :------: | :----: | :---------------------------------------------------------------------: | :--------------------: |
| YOLOv8-n |  P5  | 640  |     No      |     No     |  Yes   | Yes |   2.8    |  37.2  |       [config](../yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py)       | [model](x) \| [log](x) |
| YOLOv8-n |  P5  | 640  |     Yes     |     No     |  Yes   | Yes |    x     |   x    | [config](../yolov8/yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](x) \| [log](x) |
| YOLOv8-s |  P5  | 640  |     No      |     No     |  Yes   | Yes |   4.0    |  44.2  |       [config](../yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py)       | [model](x) \| [log](x) |
| YOLOv8-s |  P5  | 640  |     Yes     |     No     |  Yes   | Yes |    x     |   x    | [config](../yolov8/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](x) \| [log](x) |
| YOLOv8-m |  P5  | 640  |     No      |     No     |  Yes   | Yes |   7.2    |  49.8  |       [config](../yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco.py)       | [model](x) \| [log](x) |
| YOLOv8-m |  P5  | 640  |     Yes     |    Yes     |  Yes   | Yes |    x     |   x    | [config](../yolov8/yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](x) \| [log](x) |
| YOLOv8-l |  P5  | 640  |     No      |     No     |  Yes   | Yes |    x     |   x    |       [config](../yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py)       | [model](x) \| [log](x) |
| YOLOv8-l |  P5  | 640  |     Yes     |    Yes     |  Yes   | Yes |    x     |   x    | [config](../yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](x) \| [log](x) |
| YOLOv8-x |  P5  | 640  |     No      |     No     |  Yes   | Yes |    x     |   x    |       [config](../yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco.py)       | [model](x) \| [log](x) |
| YOLOv8-x |  P5  | 640  |     Yes     |    Yes     |  Yes   | Yes |    x     |   x    | [config](../yolov8/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py) | [model](x) \| [log](x) |

**Note**

1. We use 8x A100 for training, and the single-GPU batch size is 16. This is different from the official code, but has no effect on performance.
2. The performance is unstable and may fluctuate by about 0.3 mAP and the highest performance weight in `COCO` training in `YOLOv8` may not be the last epoch. The performance shown above is the best model.
3. We provide [scripts](https://github.com/open-mmlab/mmyolo/tree/dev/tools/model_converters/yolov8_to_mmyolo.py) to convert official weights to MMYOLO.
4. `SyncBN` means using SyncBN, `AMP` indicates training with mixed precision.
5. `Mask Refine` means refining bbox by mask while loading annotations and transforming after `YOLOv5RandomAffine`, `Copy Paste` means using `YOLOv5CopyPaste`.

## Citation
