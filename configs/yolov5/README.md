# YOLOv5

<!-- [ALGORITHM] -->

## Abstract

YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

## Results and models

### COCO

| Backbone | size | SyncBN | AMP | Mem (GB) | box AP |                                                        Config                                                         |                                                                                                                                                                   Download                                                                                                                                                                   |
| :------: | :--: | :----: | :-: | :------: | :----: | :-------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOv5-n | 640  |  Yes   | Yes |   1.5    |  28.0  | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739-b804c1ad.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739.log.json) |
| YOLOv5-s | 640  |  Yes   | Yes |   2.7    |  37.7  | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json) |
| YOLOv5-m | 640  |  Yes   | Yes |   5.0    |  45.3  | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944.log.json) |
| YOLOv5-l | 640  |  Yes   | Yes |   8.1    |  48.8  | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/yolov5_l-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_l-v61_syncbn_fast_8xb16-300e_coco/yolov5_l-v61_syncbn_fast_8xb16-300e_coco_20220917_031007-096ef0eb.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_l-v61_syncbn_fast_8xb16-300e_coco/yolov5_l-v61_syncbn_fast_8xb16-300e_coco_20220917_031007.log.json) |

**Note**:

1. `fast` means that `YOLOv5DetDataPreprocessor` and `yolov5_collate` are used for data preprocessing, which is faster for training, but less flexible for multitasking. Recommended to use fast version config if you only care about object detection.
2. `detect` means that the network input is fixed to `640x640` and the post-processing thresholds is modified.
3. `SyncBN` means use SyncBN, `AMP` indicates training with mixed precision.
4. We use 8x A100 for training, and the single-GPU batch size is 16. This is different from the official code.
5. The performance is unstable and may fluctuate by about 0.4 mAP.
6. `balloon` means that this is a demo configuration.

### VOC

| Backbone | size | SyncBN | AMP | Mem (GB) | box AP(COCO metric) |                                                      Config                                                      |         Download         |
| :------: | :--: | :----: | :-: | :------: | :-----------------: | :--------------------------------------------------------------------------------------------------------------: | :----------------------: |
| YOLOv5-n | 512  |   No   | Yes |   1.5    |          x          | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/voc/yolov5_n-v61_fast_1xb64-50e_voc.py) | [model](<>) \| [log](<>) |
| YOLOv5-s | 512  |   No   | Yes |   2.7    |          x          | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py) | [model](<>) \| [log](<>) |
| YOLOv5-m | 512  |   No   | Yes |   5.0    |        70.0         | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/voc/yolov5_m-v61_fast_1xb64-50e_voc.py) | [model](<>) \| [log](<>) |
| YOLOv5-l | 512  |   No   | Yes |   8.1    |          x          | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolov5/voc/yolov5_l-v61_fast_1xb32-50e_voc.py) | [model](<>) \| [log](<>) |

**Note**:

1. Training on VOC dataset need pretrained model which trained on COCO.
2. The performance is unstable and may fluctuate by about 0.4 mAP.
3. YOLOv5 use COCO metric, while training VOC dataset. We will support to use COCO metric while training VOC dataset.

## Citation

```latex
@software{glenn_jocher_2022_7002879,
  author       = {Glenn Jocher and
                  Ayush Chaurasia and
                  Alex Stoken and
                  Jirka Borovec and
                  NanoCode012 and
                  Yonghye Kwon and
                  TaoXie and
                  Kalen Michael and
                  Jiacong Fang and
                  imyhxy and
                  Lorna and
                  Colin Wong and
                  曾逸夫(Zeng Yifu) and
                  Abhiram V and
                  Diego Montes and
                  Zhiqiang Wang and
                  Cristi Fati and
                  Jebastin Nadar and
                  Laughing and
                  UnglvKitDe and
                  tkianai and
                  yxNONG and
                  Piotr Skalski and
                  Adam Hogan and
                  Max Strobel and
                  Mrinal Jain and
                  Lorenzo Mammana and
                  xylieong},
  title        = {{ultralytics/yolov5: v6.2 - YOLOv5 Classification
                   Models, Apple M1, Reproducibility, ClearML and
                   Deci.ai integrations}},
  month        = aug,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v6.2},
  doi          = {10.5281/zenodo.7002879},
  url          = {https://doi.org/10.5281/zenodo.7002879}
}
```
