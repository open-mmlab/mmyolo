# RTMDet: An Empirical Study of Designing Real-Time Object Detectors

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/real-time-instance-segmentation-on-mscoco)](https://paperswithcode.com/sota/real-time-instance-segmentation-on-mscoco?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-hrsc2016)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-hrsc2016?p=rtmdet-an-empirical-study-of-designing-real)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we aim to design an efficient real-time object detector that exceeds the YOLO series and is easily extensible for many object recognition tasks such as instance segmentation and rotated object detection. To obtain a more efficient model architecture, we explore an architecture that has compatible capacities in the backbone and neck, constructed by a basic building block that consists of large-kernel depth-wise convolutions. We further introduce soft labels when calculating matching costs in the dynamic label assignment to improve accuracy. Together with better training techniques, the resulting object detector, named RTMDet, achieves 52.8% AP on COCO with 300+ FPS on an NVIDIA 3090 GPU, outperforming the current mainstream industrial detectors. RTMDet achieves the best parameter-accuracy trade-off with tiny/small/medium/large/extra-large model sizes for various application scenarios, and obtains new state-of-the-art performance on real-time instance segmentation and rotated object detection. We hope the experimental results can provide new insights into designing versatile real-time object detectors for many object recognition tasks.

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/208070055-7233a3d8-955f-486a-82da-b714b3c3bbd6.png"/>
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/204126145-cb4ff4f1-fb16-455e-96b5-17620081023a.jpg"/>
RTMDet-l model structure
</div>

## Results and Models

### Object Detection

|    Model    | size | box AP | Params(M) | FLOPS(G) | TRT-FP16-Latency(ms) |                       Config                        |                                                                                                                                                                 Download                                                                                                                                                                 |
| :---------: | :--: | :----: | :-------: | :------: | :------------------: | :-------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RTMDet-tiny | 640  |  41.0  |    4.8    |   8.1    |         0.98         | [config](./rtmdet_l_syncbn_fast_8xb32-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117.log.json) |
|  RTMDet-s   | 640  |  44.6  |   8.89    |   14.8   |         1.22         | [config](./rtmdet_s_syncbn_fast_8xb32-300e_coco.py) |       [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-0a8c901a.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329.log.json)       |
|  RTMDet-m   | 640  |  49.3  |   24.71   |  39.27   |         1.62         | [config](./rtmdet_m_syncbn_fast_8xb32-300e_coco.py) |       [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952.log.json)       |
|  RTMDet-l   | 640  |  51.4  |   52.3    |  80.23   |         2.44         | [config](./rtmdet_l_syncbn_fast_8xb32-300e_coco.py) |       [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928.log.json)       |
|  RTMDet-x   | 640  |  52.8  |   94.86   |  141.67  |         3.10         | [config](./rtmdet_x_syncbn_fast_8xb32-300e_coco.py) |       [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_x_syncbn_fast_8xb32-300e_coco/rtmdet_x_syncbn_fast_8xb32-300e_coco_20221231_100345-b85cd476.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_x_syncbn_fast_8xb32-300e_coco/rtmdet_x_syncbn_fast_8xb32-300e_coco_20221231_100345.log.json)       |

**Note**:

1. The inference speed of RTMDet is measured on an NVIDIA 3090 GPU with TensorRT 8.4.3, cuDNN 8.2.0, FP16, batch size=1, and without NMS.
2. For a fair comparison, the config of bbox postprocessing is changed to be consistent with YOLOv5/6/7 after [PR#9494](https://github.com/open-mmlab/mmdetection/pull/9494), bringing about 0.1~0.3% AP improvement.

### Rotated Object Detection

RTMDet-R achieves state-of-the-art on various remote sensing datasets.

|  Backbone   | pretrain |       Aug       | mmAP  | mAP50 | mAP75 | Params(M) | FLOPS(G) | TRT-FP16-Latency(ms) |                                 Config                                 |         Download         |
| :---------: | :------: | :-------------: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :--------------------------------------------------------------------: | :----------------------: |
| RTMDet-tiny |    IN    |       RR        | 46.81 | 75.88 | 50.08 |   4.88    |  20.45   |         4.40         |      [config](./rotate/rtmdet_tiny_syncbn_fast_1xb8-36e_dota.py)       | [model](<>) \| [log](<>) |
|  RTMDet-s   |    IN    |       RR        | 48.22 | 77.09 | 50.61 |   8.86    |  37.62   |         4.86         |        [config](./rotate/rtmdet_s_syncbn_fast_1xb8-36e_dota.py)        | [model](<>) \| [log](<>) |
|  RTMDet-m   |    IN    |       RR        |   -   |   -   |   -   |   24.67   |  99.76   |         7.82         |        [config](./rotate/rtmdet_m_syncbn_fast_1xb8-36e_dota.py)        | [model](<>) \| [log](<>) |
|  RTMDet-l   |    IN    |       RR        |   -   |   -   |   -   |   52.27   |  204.21  |        10.82         |        [config](./rotate/rtmdet_l_syncbn_fast_1xb8-36e_dota.py)        | [model](<>) \| [log](<>) |
| RTMDet-tiny |    IN    |      MS+RR      |   -   |   -   |   -   |   4.88    |  20.45   |         4.40         |     [config](./rotate/rtmdet_tiny_syncbn_fast_1xb8-36e_dota_ms.py)     | [model](<>) \| [log](<>) |
|  RTMDet-s   |    IN    |      MS+RR      |   -   |   -   |   -   |   8.86    |  37.62   |         4.86         |      [config](./rotate/rtmdet_s_syncbn_fast_1xb8-36e_dota_ms.py)       | [model](<>) \| [log](<>) |
|  RTMDet-m   |    IN    |      MS+RR      |   -   |   -   |   -   |   24.67   |  99.76   |         7.82         |      [config](./rotate/rtmdet_m_syncbn_fast_1xb8-36e_dota_ms.py)       | [model](<>) \| [log](<>) |
|  RTMDet-l   |    IN    |      MS+RR      |   -   |   -   |   -   |   52.27   |  204.21  |        10.82         |      [config](./rotate/rtmdet_l_syncbn_fast_1xb8-36e_dota_ms.py)       | [model](<>) \| [log](<>) |
|  RTMDet-l   |   COCO   |      MS+RR      |   -   |   -   |   -   |   52.27   |  204.21  |        10.82         | [config](./rotate/rtmdet-r_l_pretrain_syncbn_fast_1xb8_36e_dota_ms.py) | [model](<>) \| [log](<>) |
|  RTMDet-l   |    IN    | Mixup+Mosaic+RR |   -   |   -   |   -   |   52.27   |  204.21  |        10.82         |      [config](./rotate/rtmdet-r_l_syncbn_fast_1xb8_100e_dota.py)       | [model](<>) \| [log](<>) |

**Note**:

1. Please follow doc to prerare data first. (TODO)
2. We follow the latest metrics from the DOTA evaluation server, original voc format mAP is now mAP50.
3. All models trained with image size 1024\*1024.
4. `IN` means ImageNet pretrain, `COCO` means COCO pretrain.
5. For Aug, RR means `RandomRotate`, MS means multi-scale augmentation in data prepare.

## Citation

```latex
@misc{lyu2022rtmdet,
      title={RTMDet: An Empirical Study of Designing Real-Time Object Detectors},
      author={Chengqi Lyu and Wenwei Zhang and Haian Huang and Yue Zhou and Yudong Wang and Yanyi Liu and Shilong Zhang and Kai Chen},
      year={2022},
      eprint={2212.07784},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
