# YOLOv7

> [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

<!-- [ALGORITHM] -->

## Abstract

YOLOv7 surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS and has the highest accuracy 56.8% AP among all known real-time object detectors with 30 FPS or higher on GPU V100. YOLOv7-E6 object detector (56 FPS V100, 55.9% AP) outperforms both transformer-based detector SWIN-L Cascade-Mask R-CNN (9.2 FPS A100, 53.9% AP) by 509% in speed and 2% in accuracy, and convolutional-based detector ConvNeXt-XL Cascade-Mask R-CNN (8.6 FPS A100, 55.2% AP) by 551% in speed and 0.7% AP in accuracy, as well as YOLOv7 outperforms: YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5, DETR, Deformable DETR, DINO-5scale-R50, ViT-Adapter-B and many other object detectors in speed and accuracy. Moreover, we train YOLOv7 only on MS COCO dataset from scratch without using any other datasets or pre-trained weights. Source code is released in [this https URL](https://github.com/WongKinYiu/yolov7).

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/204231759-cc5c77a9-38c6-4a41-85be-eb97e4b2bcbb.png"/>
</div>

## Results and models

### COCO

| Backbone | Arch | Size | SyncBN | AMP | Mem (GB) | Box AP |                           Config                            |                                                                                                                                                           Download                                                                                                                                                           |
| :------: | :--: | :--: | :---: | :----: | :-: | :------: | :----: | :---------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOv7-tiny |  P5  | 640  | Yes   | Yes |   xx   |  xx  | [config](../yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_n_syncbn_fast_8xb32-400e_coco/yolov6_n_syncbn_fast_8xb32-400e_coco_20221030_202726-d99b2e82.pth) | [log](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_n_syncbn_fast_8xb32-400e_coco/yolov6_n_syncbn_fast_8xb32-400e_coco_20221030_202726.log.json) |
| YOLOv7-l |  P5  | 640  |   Yes   | Yes |   xx   |  xx  | [config](../yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_t_syncbn_fast_8xb32-400e_coco/yolov6_t_syncbn_fast_8xb32-400e_coco_20221030_143755-cf0d278f.pth) | [log](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_t_syncbn_fast_8xb32-400e_coco/yolov6_t_syncbn_fast_8xb32-400e_coco_20221030_143755.log.json) |
| YOLOv7-x |  P5  | 640  |   Yes   | Yes |   xx   |  xx  | [config](../yolov7/yolov7_x_syncbn_fast_8x16b-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth) | [log](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035.log.json) |
| YOLOv7-w |  P6  | 1280  |   Yes   | Yes |  xx   |  xx  | [config](../yolov7/yolov7_w-p6_syncbn_fast_8x16b-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_m_syncbn_fast_8xb32-300e_coco/yolov6_m_syncbn_fast_8xb32-300e_coco_20221109_182658-85bda3f4.pth) | [log](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_m_syncbn_fast_8xb32-300e_coco/yolov6_m_syncbn_fast_8xb32-300e_coco_20221109_182658.log.json) |
| YOLOv7-d |  P6  | 1280  |   Yes   | Yes |  xx   |  xx  | [config](../yolov7/yolov7_d-p6_syncbn_fast_8x16b-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_l_syncbn_fast_8xb32-300e_coco/yolov6_l_syncbn_fast_8xb32-300e_coco_20221109_183156-91e3c447.pth) | [log](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_l_syncbn_fast_8xb32-300e_coco/yolov6_l_syncbn_fast_8xb32-300e_coco_20221109_183156.log.json) |
| YOLOv7-e |  P6  | 1280  |   Yes   | Yes |  xx   |  xx  | [config](../yolov7/yolov7_e-p6_syncbn_fast_8x16b-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_l_syncbn_fast_8xb32-300e_coco/yolov6_l_syncbn_fast_8xb32-300e_coco_20221109_183156-91e3c447.pth) | [log](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_l_syncbn_fast_8xb32-300e_coco/yolov6_l_syncbn_fast_8xb32-300e_coco_20221109_183156.log.json) |

**Note**:
In the official YOLOv7 code, the `random_perspective` data augmentation in COCO object detection task training uses mask annotation information, which leads to higher performance. Object detection should not use mask annotation, so only box annotation information is used in `MMYOLO`. We will use the mask annotation information in the instance segmentation task.

1. The performance is unstable and may fluctuate by about 0.3 mAP.
2. If users need the weight of `YOLOv7-e2e`, they can train according to the configs provided by us, or convert the official weight according to the [converter script](../../tools/model_converters/yolov7_to_mmyolo.py).
3. `fast` means that `YOLOv5DetDataPreprocessor` and `yolov5_collate` are used for data preprocessing, which is faster for training, but less flexible for multitasking. Recommended to use fast version config if you only care about object detection.
4. `SyncBN` means use SyncBN, `AMP` indicates training with mixed precision.
5. We use 8x A100 for training, and the single-GPU batch size is 16. This is different from the official code.

## Citation

```latex
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```
