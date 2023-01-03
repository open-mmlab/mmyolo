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

## Results and Models

## Object Detection

|    Model    | size | box AP | Params(M) | FLOPS(G) | TRT-FP16-Latency(ms) |                       Config                        |         Download         |
| :---------: | :--: | :----: | :-------: | :------: | :------------------: | :-------------------------------------------------: | :----------------------: |
| RTMDet-tiny | 640  |  41.1  |    4.8    |   8.1    |         0.98         | [config](./rtmdet_l_syncbn_fast_8xb32-300e_coco.py) | [model](<>) \| [log](<>) |
|  RTMDet-s   | 640  |  44.6  |   8.89    |   14.8   |         1.22         | [config](./rtmdet_s_syncbn_fast_8xb32-300e_coco.py) | [model](<>) \| [log](<>) |
|  RTMDet-m   | 640  |  49.4  |   24.71   |  39.27   |         1.62         | [config](./rtmdet_m_syncbn_fast_8xb32-300e_coco.py) | [model](<>) \| [log](<>) |
|  RTMDet-l   | 640  |  51.5  |   52.3    |  80.23   |         2.44         | [config](./rtmdet_l_syncbn_fast_8xb32-300e_coco.py) | [model](<>) \| [log](<>) |
|  RTMDet-x   | 640  |  52.8  |   94.86   |  141.67  |         3.10         | [config](./rtmdet_x_syncbn_fast_8xb32-300e_coco.py) | [model](<>) \| [log](<>) |

**Note**:

1. The inference speed of RTMDet is measured on an NVIDIA 3090 GPU with TensorRT 8.4.3, cuDNN 8.2.0, FP16, batch size=1, and without NMS.
2. For a fair comparison, the config of bbox postprocessing is changed to be consistent with YOLOv5/6/7 after [PR#9494](https://github.com/open-mmlab/mmdetection/pull/9494), bringing about 0.1~0.3% AP improvement.
