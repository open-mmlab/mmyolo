# YOLOX

> [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)

<!-- [ALGORITHM] -->

## Abstract

In this report, we present some experienced improvements to YOLO series, forming a new high-performance detector -- YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale range of models: For YOLO-Nano with only 0.91M parameters and 1.08G FLOPs, we get 25.3% AP on COCO, surpassing NanoDet by 1.8% AP; for YOLOv3, one of the most widely used detectors in industry, we boost it to 47.3% AP on COCO, outperforming the current best practice by 3.0% AP; for YOLOX-L with roughly the same amount of parameters as YOLOv4-CSP, YOLOv5-L, we achieve 50.0% AP on COCO at a speed of 68.9 FPS on Tesla V100, exceeding YOLOv5-L by 1.8% AP. Further, we won the 1st Place on Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021) using a single YOLOX-L model. We hope this report can provide useful experience for developers and researchers in practical scenes, and we also provide deploy versions with ONNX, TensorRT, NCNN, and Openvino supported.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/144001736-9fb303dd-eac7-46b0-ad45-214cfa51e928.png"/>
</div>
<div align=center>
<img src="https://user-images.githubusercontent.com/71306851/208933940-ffcd2f53-7630-4c7e-bf0d-695eb3cfe851.jpg"/>
YOLOX_l model structure
</div>

## Results and Models

|  Backbone  | size | Mem (GB) | box AP |                                                Config                                                 |                                                                                                                                    Download                                                                                                                                    |
| :--------: | :--: | :------: | :----: | :---------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOX-tiny | 416  |   2.8    |  32.7  | [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolox/yolox_tiny_8xb8-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_8xb8-300e_coco/yolox_tiny_8xb8-300e_coco_20220919_090908-0e40a6fc.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_8xb8-300e_coco/yolox_tiny_8xb8-300e_coco_20220919_090908.log.json) |
|  YOLOX-s   | 640  |   5.6    |  40.8  |  [config](https://github.com/open-mmlab/mmyolo/tree/master/configs/yolox/yolox_s_8xb8-300e_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20220917_030738-d7e60cb2.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20220917_030738.log.json)       |

**Note**:

1. The test score threshold is 0.001.
2. Due to the need for pre-training weights, we cannot reproduce the performance of the `yolox-nano` model. Please refer to https://github.com/Megvii-BaseDetection/YOLOX/issues/674 for more information.

## Citation

```latex
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
