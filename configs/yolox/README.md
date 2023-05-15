# YOLOX

> [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)

<!-- [ALGORITHM] -->

## Abstract

In this report, we present some experienced improvements to YOLO series, forming a new high-performance detector -- YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale range of models: For YOLO-Nano with only 0.91M parameters and 1.08G FLOPs, we get 25.3% AP on COCO, surpassing NanoDet by 1.8% AP; for YOLOv3, one of the most widely used detectors in industry, we boost it to 47.3% AP on COCO, outperforming the current best practice by 3.0% AP; for YOLOX-L with roughly the same amount of parameters as YOLOv4-CSP, YOLOv5-L, we achieve 50.0% AP on COCO at a speed of 68.9 FPS on Tesla V100, exceeding YOLOv5-L by 1.8% AP. Further, we won the 1st Place on Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021) using a single YOLOX-L model. We hope this report can provide useful experience for developers and researchers in practical scenes, and we also provide deploy versions with ONNX, TensorRT, NCNN, and Openvino supported.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/144001736-9fb303dd-eac7-46b0-ad45-214cfa51e928.png"/>
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/71306851/218628641-6c0101e6-e40e-4b16-a696-c0f55b8d335c.png"/>
YOLOX-l model structure
</div>

## 🥳 🚀 Results and Models

|  Backbone  | Size | Batch Size | AMP | RTMDet-Hyp | Mem (GB) |   Box AP    |                          Config                           |                                                                                                                                                                      Download                                                                                                                                                                      |
| :--------: | :--: | :--------: | :-: | :--------: | :------: | :---------: | :-------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOX-tiny | 416  |    8xb8    | No  |     No     |   2.8    |    32.7     |       [config](./yolox_tiny_fast_8xb8-300e_coco.py)       |                                   [model](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_8xb8-300e_coco/yolox_tiny_8xb8-300e_coco_20220919_090908-0e40a6fc.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_8xb8-300e_coco/yolox_tiny_8xb8-300e_coco_20220919_090908.log.json)                                   |
| YOLOX-tiny | 416  |   8xb32    | Yes |    Yes     |   4.9    | 34.3 (+1.6) | [config](./yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco/yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco_20230210_143637-4c338102.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco/yolox_tiny_fast_8xb32-300e-rtmdet-hyp_coco_20230210_143637.log.json) |
|  YOLOX-s   | 640  |    8xb8    | Yes |     No     |   2.9    |    40.7     |        [config](./yolox_s_fast_8xb8-300e_coco.py)         |                               [model](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_fast_8xb8-300e_coco/yolox_s_fast_8xb8-300e_coco_20230213_142600-2b224d8b.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_fast_8xb8-300e_coco/yolox_s_fast_8xb8-300e_coco_20230213_142600.log.json)                               |
|  YOLOX-s   | 640  |   8xb32    | Yes |    Yes     |   9.8    | 41.9 (+1.2) |  [config](./yolox_s_fast_8xb32-300e-rtmdet-hyp_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco_20230210_134645-3a8dfbd7.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco_20230210_134645.log.json)       |
|  YOLOX-m   | 640  |    8xb8    | Yes |     No     |   4.9    |    46.9     |        [config](./yolox_m_fast_8xb8-300e_coco.py)         |                               [model](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_m_fast_8xb8-300e_coco/yolox_m_fast_8xb8-300e_coco_20230213_160218-a71a6b25.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_m_fast_8xb8-300e_coco/yolox_m_fast_8xb8-300e_coco_20230213_160218.log.json)                               |
|  YOLOX-m   | 640  |   8xb32    | Yes |    Yes     |   17.6   | 47.5 (+0.6) |  [config](./yolox_m_fast_8xb32-300e-rtmdet-hyp_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_m_fast_8xb32-300e-rtmdet-hyp_coco/yolox_m_fast_8xb32-300e-rtmdet-hyp_coco_20230210_144328-e657e182.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_m_fast_8xb32-300e-rtmdet-hyp_coco/yolox_m_fast_8xb32-300e-rtmdet-hyp_coco_20230210_144328.log.json)       |
|  YOLOX-l   | 640  |    8xb8    | Yes |     No     |   8.0    |    50.1     |        [config](./yolox_l_fast_8xb8-300e_coco.py)         |                               [model](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_l_fast_8xb8-300e_coco/yolox_l_fast_8xb8-300e_coco_20230213_160715-c731eb1c.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_l_fast_8xb8-300e_coco/yolox_l_fast_8xb8-300e_coco_20230213_160715.log.json)                               |
|  YOLOX-x   | 640  |    8xb8    | Yes |     No     |   9.8    |    51.4     |        [config](./yolox_x_fast_8xb8-300e_coco.py)         |                               [model](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_x_fast_8xb8-300e_coco/yolox_x_fast_8xb8-300e_coco_20230215_133950-1d509fab.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/yolox_x_fast_8xb8-300e_coco/yolox_x_fast_8xb8-300e_coco_20230215_133950.log.json)                               |

YOLOX uses a default training configuration of `8xbs8` which results in a long training time, we expect it to use `8xbs32` to speed up the training and not cause a decrease in mAP. We modified `train_batch_size_per_gpu` from 8 to 32, `batch_augments_interval` from 10 to 1 and `base_lr` from 0.01 to 0.04 under YOLOX-s default configuration based on the linear scaling rule, which resulted in mAP degradation. Finally, I found that using RTMDet's training hyperparameter can improve performance in YOLOX Tiny/S/M, which also validates the superiority of RTMDet's training hyperparameter.

The modified training parameters are as follows：

1. train_batch_size_per_gpu: 8 -> 32
2. batch_augments_interval: 10 -> 1
3. num_last_epochs: 15 -> 20
4. optim cfg: SGD -> AdamW, base_lr 0.01 -> 0.004, weight_decay 0.0005 -> 0.05
5. ema momentum: 0.0001 -> 0.0002

**Note**:

1. The test score threshold is 0.001.
2. Due to the need for pre-training weights, we cannot reproduce the performance of the `yolox-nano` model. Please refer to https://github.com/Megvii-BaseDetection/YOLOX/issues/674 for more information.

## YOLOX-Pose

Based on [MMPose](https://github.com/open-mmlab/mmpose/blob/main/projects/yolox-pose/README.md), we have implemented a YOLOX-based human pose estimator, utilizing the approach outlined in **YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss (CVPRW 2022)**. This pose estimator is lightweight and quick, making it well-suited for crowded scenes.

<div align=center>
<img src="https://user-images.githubusercontent.com/26127467/226655503-3cee746e-6e42-40be-82ae-6e7cae2a4c7e.jpg"/>
</div>

### Results

|  Backbone  | Size | Batch Size | AMP | RTMDet-Hyp | Mem (GB) |  AP  |                             Config                             |                                                                                                                                                                           Download                                                                                                                                                                           |
| :--------: | :--: | :--------: | :-: | :--------: | :------: | :--: | :------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOX-tiny | 416  |   8xb32    | Yes |    Yes     |   5.3    | 52.8 | [config](./pose/yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco/yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco_20230427_080351-2117af67.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco/yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco_20230427_080351.log.json) |
|  YOLOX-s   | 640  |   8xb32    | Yes |    Yes     |   10.7   | 63.7 |  [config](./pose/yolox-pose_s_8xb32-300e-rtmdet-hyp_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_s_8xb32-300e-rtmdet-hyp_coco/yolox-pose_s_8xb32-300e-rtmdet-hyp_coco_20230427_005150-e87d843a.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_s_8xb32-300e-rtmdet-hyp_coco/yolox-pose_s_8xb32-300e-rtmdet-hyp_coco_20230427_005150.log.json)       |
|  YOLOX-m   | 640  |   8xb32    | Yes |    Yes     |   19.2   | 69.3 |  [config](./pose/yolox-pose_m_8xb32-300e-rtmdet-hyp_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_m_8xb32-300e-rtmdet-hyp_coco/yolox-pose_m_8xb32-300e-rtmdet-hyp_coco_20230427_094024-bbeacc1c.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_m_8xb32-300e-rtmdet-hyp_coco/yolox-pose_m_8xb32-300e-rtmdet-hyp_coco_20230427_094024.log.json)       |
|  YOLOX-l   | 640  |   8xb32    | Yes |    Yes     |   30.3   | 71.1 |  [config](./pose/yolox-pose_l_8xb32-300e-rtmdet-hyp_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_l_8xb32-300e-rtmdet-hyp_coco/yolox-pose_l_8xb32-300e-rtmdet-hyp_coco_20230427_041140-82d65ac8.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_l_8xb32-300e-rtmdet-hyp_coco/yolox-pose_l_8xb32-300e-rtmdet-hyp_coco_20230427_041140.log.json)       |

**Note**

1. The performance is unstable and may fluctuate and the highest performance weight in `COCO` training may not be the last epoch. The performance shown above is the best model.

### Installation

Install MMPose

```
mim install -r requirements/mmpose.txt
```

## Citation

```latex
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
