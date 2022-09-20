# YOLOv5

<!-- [ALGORITHM] -->

## Abstract

YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

## Results and models

### COCO

| Backbone | size | SyncBN | AMP | Mem (GB) | box AP |                                                        Config                                                         |                                                                                                                                                                   Download                                                                                                                                                                    |
| :------: | :--: | :----: | :-: | :------: | :----: | :-------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOv5-n | 640  |  Yes   | Yes |   1.5    |  28.0  | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739-b804c1ad.pthh) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739.log.json) |
| YOLOv5-s | 640  |  Yes   | Yes |   2.7    |  37.7  | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json)  |
| YOLOv5-m | 640  |  Yes   | Yes |   5.0    |  45.3  | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944.log.json)  |
| YOLOv5-l | 640  |  Yes   | Yes |   8.1    |  48.8  | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/yolov5_l-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_l-v61_syncbn_fast_8xb16-300e_coco/yolov5_l-v61_syncbn_fast_8xb16-300e_coco_20220917_031007-096ef0eb.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_l-v61_syncbn_fast_8xb16-300e_coco/yolov5_l-v61_syncbn_fast_8xb16-300e_coco_20220917_031007.log.json)  |

**Note**:

1. `fast` means that `YOLOv5DetDataPreprocessor` and `yolov5_collate` are used for data preprocessing, which is faster for training, but less flexible for multitasking. Recommended to use fast version config if you only care about object detection.
2. `SyncBN` means use SyncBN, `AMP` indicates training with mixed precision.
3. We use 8x A100 for training, and the single-GPU batch size is 16. This is different from the official code.
4. The performance is unstable and may fluctuate by about 0.4 mAP.
