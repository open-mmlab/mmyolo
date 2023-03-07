# PPYOLOE

<!-- [ALGORITHM] -->

## Abstract

PP-YOLOE is an excellent single-stage anchor-free model based on PP-YOLOv2, surpassing a variety of popular YOLO models. PP-YOLOE has a series of models, named s/m/l/x, which are configured through width multiplier and depth multiplier. PP-YOLOE avoids using special operators, such as Deformable Convolution or Matrix NMS, to be deployed friendly on various hardware.

<div align=center>
<img src="https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/images/ppyoloe_plus_map_fps.png" width="600" />
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/71306851/213100232-a2e278a6-0b97-4d21-9c1b-09eabb741b84.png"/>
PPYOLOE-PLUS-l model structure
</div>

## Results and models

### PPYOLOE+ COCO

|  Backbone   | Arch | Size | Epoch | SyncBN | Mem (GB) | Box AP |                      Config                      |                                                                                                                                                      Download                                                                                                                                                      |
| :---------: | :--: | :--: | :---: | :----: | :------: | :----: | :----------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| PPYOLOE+ -s |  P5  | 640  |  80   |  Yes   |   4.7    |  43.5  | [config](./ppyoloe_plus_s_fast_8xb8-80e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco/ppyoloe_plus_s_fast_8xb8-80e_coco_20230101_154052-9fee7619.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco/ppyoloe_plus_s_fast_8xb8-80e_coco_20230101_154052.log.json) |
| PPYOLOE+ -m |  P5  | 640  |  80   |  Yes   |   8.4    |  49.5  | [config](./ppyoloe_plus_m_fast_8xb8-80e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_m_fast_8xb8-80e_coco/ppyoloe_plus_m_fast_8xb8-80e_coco_20230104_193132-e4325ada.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_m_fast_8xb8-80e_coco/ppyoloe_plus_m_fast_8xb8-80e_coco_20230104_193132.log.json) |
| PPYOLOE+ -l |  P5  | 640  |  80   |  Yes   |   13.2   |  52.6  | [config](./ppyoloe_plus_l_fast_8xb8-80e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_l_fast_8xb8-80e_coco/ppyoloe_plus_l_fast_8xb8-80e_coco_20230102_203825-1864e7b3.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_l_fast_8xb8-80e_coco/ppyoloe_plus_l_fast_8xb8-80e_coco_20230102_203825.log.json) |
| PPYOLOE+ -x |  P5  | 640  |  80   |  Yes   |   19.1   |  54.2  | [config](./ppyoloe_plus_x_fast_8xb8-80e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_x_fast_8xb8-80e_coco/ppyoloe_plus_x_fast_8xb8-80e_coco_20230104_194921-8c953949.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_x_fast_8xb8-80e_coco/ppyoloe_plus_x_fast_8xb8-80e_coco_20230104_194921.log.json) |

**Note**:

1. The above Box APs are all models with the best performance in COCO
2. The gap between the above performance and the official release is about 0.3. To speed up training in mmyolo, we use pytorch to implement the image resizing in `PPYOLOEBatchRandomResize` for multi-scale training, while official PPYOLOE use opencv. And `lanczos4` is not yet supported in `PPYOLOEBatchRandomResize`. The above two reasons lead to the gap. We will continue to experiment and address the gap in future releases.
3. The mAP of the non-Plus version needs more verification, and we will update more details of the non-Plus version in future versions.

```latex
@article{Xu2022PPYOLOEAE,
  title={PP-YOLOE: An evolved version of YOLO},
  author={Shangliang Xu and Xinxin Wang and Wenyu Lv and Qinyao Chang and Cheng Cui and Kaipeng Deng and Guanzhong Wang and Qingqing Dang and Shengyun Wei and Yuning Du and Baohua Lai},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.16250}
}
```
