# RTMDet

<!-- [ALGORITHM] -->

## Abstract

Our tech-report will be released soon.

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/192182907-f9a671d6-89cb-4d73-abd8-c2b9dada3c66.png"/>
</div>

## Results and Models

|  Backbone   | size | SyncBN | box AP | Params(M) | FLOPS(G) | TRT-FP16-Latency(ms) |                      Config                       |                                                                                                                                                   Download                                                                                                                                                    |
| :---------: | :--: | :----: | -----: | :-------: | :------: | :------------------: | :-----------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| RTMDet-tiny | 640  |  Yes   |   40.9 |    4.8    |   8.1    |         0.98         | [config](./rtmdet_tiny_syncbn_8xb32-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_8xb32-300e_coco/rtmdet_tiny_syncbn_8xb32-300e_coco_20220902_112414-259f3241.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414.log.json) |
|  RTMDet-s   | 640  |  Yes   |   44.5 |   8.89    |   14.8   |         1.22         |  [config](./rtmdet_s_syncbn_8xb32-300e_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_s_syncbn_8xb32-300e_coco/rtmdet_s_syncbn_8xb32-300e_coco_20220905_161602-fd1cacb9.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602.log.json)       |
|  RTMDet-m   | 640  |  Yes   |   49.1 |   24.71   |  39.27   |         1.62         |  [config](./rtmdet_m_syncbn_8xb32-300e_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_m_syncbn_8xb32-300e_coco/rtmdet_m_syncbn_8xb32-300e_coco_20220924_132959-d9f2e90d.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220924_132959.log.json)       |
|  RTMDet-l   | 640  |  Yes   |   51.3 |   52.3    |  80.23   |         2.44         |  [config](./rtmdet_l_syncbn_8xb32-300e_coco.py)   |       [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_l_syncbn_8xb32-300e_coco/rtmdet_l_syncbn_8xb32-300e_coco_20220926_150401-40c754b5.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220926_150401.log.json)       |
|  RTMDet-x   | 640  |  Yes   |   52.6 |   94.86   |  141.67  |         3.10         |  [config](./rtmdet_x_syncbn_8xb32-300e_coco.py)   |                                                                                                                                           [model](<>) \| [log](<>)                                                                                                                                            |

**Note**:

1. The inference speed is measured on an NVIDIA 3090 GPU with TensorRT 8.4.3, cuDNN 8.2.0, FP16, batch size=1, and without NMS.
2. We still directly use the weights trained by `mmdet` currently. A re-trained model will be released later.
