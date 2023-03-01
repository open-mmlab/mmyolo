# Model Zoo and Benchmark

This page is used to summarize the performance and related evaluation metrics of various models supported in MMYOLO for users to compare and analyze.

## COCO dataset

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/222087414-168175cc-dae6-4c5c-a8e3-3109a152dd19.png"/>
</div>

|      Model       | Arch | Size | Batch Size | Epoch | SyncBN | AMP | Mem (GB) | Params(M) | FLOPs(G) | TRT-FP16-GPU-Latency(ms) | Box AP | TTA Box AP |
| :--------------: | :--: | :--: | :--------: | :---: | :----: | :-: | :------: | :-------: | :------: | :----------------------: | :----: | :--------: |
|     YOLOv5-n     |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   1.5    |   1.87    |   2.26   |           1.14           |  28.0  |    30.7    |
|  YOLOv6-v2.0-n   |  P5  | 640  |   8xb32    |  400  |  Yes   | Yes |   6.04   |   4.32    |   5.52   |           1.37           |  36.2  |            |
|     YOLOv8-n     |  P5  | 640  |   8xb16    |  500  |  Yes   | Yes |   2.5    |   3.16    |   4.4    |           1.53           |  37.4  |    39.9    |
|   RTMDet-tiny    |  P5  | 640  |   8xb32    |  300  |  Yes   | No  |   11.9   |   4.90    |   8.09   |           2.31           |  41.8  |    43.2    |
| YOLOv6-v2.0-tiny |  P5  | 640  |   8xb32    |  400  |  Yes   | Yes |   8.13   |   9.70    |  12.37   |           2.19           |  41.0  |            |
|   YOLOv7-tiny    |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   2.7    |   6.23    |   6.89   |           1.88           |  37.5  |            |
|    YOLOX-tiny    |  P5  | 416  |   8xb32    |  300  |   No   | Yes |   4.9    |   5.06    |   7.63   |           1.19           |  34.3  |            |
|     RTMDet-s     |  P5  | 640  |   8xb32    |  300  |  Yes   | No  |   16.3   |   8.89    |  14.84   |           2.89           |  45.7  |    47.3    |
|     YOLOv5-s     |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   2.7    |   7.24    |   8.27   |           1.89           |  37.7  |    40.2    |
|  YOLOv6-v2.0-s   |  P5  | 640  |   8xb32    |  400  |  Yes   | Yes |   8.88   |   17.22   |  21.94   |           2.67           |  44.0  |            |
|     YOLOv8-s     |  P5  | 640  |   8xb16    |  500  |  Yes   | Yes |   4.0    |   11.17   |  14.36   |           2.61           |  45.1  |    46.8    |
|     YOLOX-s      |  P5  | 640  |   8xb32    |  300  |   No   | Yes |   9.8    |   8.97    |  13.40   |           2.38           |  41.9  |            |
|   PPYOLOE+ -s    |  P5  | 640  |    8xb8    |  80   |  Yes   | No  |   4.7    |   7.93    |   8.68   |           2.54           |  43.5  |            |
|     RTMDet-m     |  P5  | 640  |   8xb32    |  300  |  Yes   | No  |   29.0   |   24.71   |  39.21   |           6.23           |  50.2  |    51.9    |
|     YOLOv5-m     |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   5.0    |   21.19   |  24.53   |           4.28           |  45.3  |    46.9    |
|  YOLOv6-v2.0-m   |  P5  | 640  |   8xb32    |  300  |  Yes   | Yes |  16.69   |   34.25   |   40.7   |           5.12           |  48.4  |            |
|     YOLOv8-m     |  P5  | 640  |   8xb16    |  500  |  Yes   | Yes |   7.0    |   25.9    |  39.57   |           5.78           |  50.6  |    52.3    |
|     YOLOX-m      |  P5  | 640  |   8xb32    |  300  |   No   | Yes |   17.6   |   25.33   |  36.88   |           5.31           |  47.5  |            |
|   PPYOLOE+ -m    |  P5  | 640  |    8xb8    |  80   |  Yes   | No  |   8.4    |   23.43   |  24.97   |           5.47           |  49.5  |            |
|     RTMDet-l     |  P5  | 640  |   8xb32    |  300  |  Yes   | No  |   45.2   |   52.32   |  80.12   |          10.13           |  52.3  |    53.7    |
|     YOLOv5-l     |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   8.1    |   46.56   |  54.65   |           6.8            |  48.8  |    49.9    |
|  YOLOv6-v2.0-l   |  P5  | 640  |   8xb32    |  300  |  Yes   | Yes |  20.86   |   58.53   |  71.43   |           8.78           |  51.0  |            |
|     YOLOv7-l     |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   10.3   |   36.93   |  52.42   |           6.63           |  50.9  |            |
|     YOLOv8-l     |  P5  | 640  |   8xb16    |  500  |  Yes   | Yes |   9.1    |   43.69   |  82.73   |           8.97           |  53.0  |    54.4    |
|     YOLOX-l      |  P5  | 640  |    8xb8    |  300  |   No   | Yes |   8.0    |   54.21   |  77.83   |           9.23           |  50.1  |            |
|   PPYOLOE+ -l    |  P5  | 640  |    8xb8    |  80   |  Yes   | No  |   13.2   |   52.20   |  55.05   |           8.2            |  52.6  |            |
|     RTMDet-x     |  P5  | 640  |   8xb32    |  300  |  Yes   | No  |   63.4   |   94.86   |  145.41  |          17.89           |  52.8  |    54.2    |
|     YOLOv7-x     |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   13.7   |   71.35   |  95.06   |          11.63           |  52.8  |            |
|     YOLOv8-x     |  P5  | 640  |   8xb16    |  500  |  Yes   | Yes |   12.4   |   68.23   |  132.10  |          14.22           |  54.0  |    55.0    |
|     YOLOX-x      |  P5  | 640  |    8xb8    |  300  |   No   | Yes |   9.8    |   99.07   |  144.39  |          15.35           |  51.4  |            |
|   PPYOLOE+ -x    |  P5  | 640  |    8xb8    |  80   |  Yes   | No  |   19.1   |   98.42   |  105.48  |          14.02           |  54.2  |            |
|     YOLOv5-n     |  P6  | 1280 |   8xb16    |  300  |  Yes   | Yes |   5.8    |   3.25    |   2.30   |                          |  35.9  |            |
|     YOLOv5-s     |  P6  | 1280 |   8xb16    |  300  |  Yes   | Yes |   10.5   |   12.63   |   8.45   |                          |  44.4  |            |
|     YOLOv5-m     |  P6  | 1280 |   8xb16    |  300  |  Yes   | Yes |   19.1   |   35.73   |  25.05   |                          |  51.3  |            |
|     YOLOv5-l     |  P6  | 1280 |   8xb16    |  300  |  Yes   | Yes |   30.5   |   76.77   |  55.77   |                          |  53.7  |            |
|     YOLOv7-w     |  P6  | 1280 |   8xb16    |  300  |  Yes   | Yes |   27.0   |   82.31   |  45.07   |                          |  54.1  |            |
|     YOLOv7-e     |  P6  | 1280 |   8xb16    |  300  |  Yes   | Yes |   42.5   |  114.69   |  64.48   |                          |  55.1  |            |

- All the models are trained on COCO train2017 dataset and evaluated on val2017 dataset.
- TRT-FP16-GPU-Latency(ms) is the GPU Compute time on NVIDIA Tesla T4 device with TensorRT 8.4, a batch size of 1, a test shape of 640x640 and only model forward (The test shape for YOLOX-tiny is 416x416)
- The number of model parameters and FLOPs are obtained using the [get_flops](https://github.com/open-mmlab/mmyolo/blob/dev/tools/analysis_tools/get_flops.py) script. Different calculation methods may vary slightly
- RTMDet performance is the result of training with [MMRazor Knowledge Distillation](https://github.com/open-mmlab/mmyolo/blob/dev/configs/rtmdet/distillation/README.md)
- Only YOLOv6 version 2.0 is implemented in MMYOLO for now, and L and M are the results without knowledge distillation
- YOLOv8 results are optimized using mask instance annotation, but YOLOv5, YOLOv6 and YOLOv7 do not use
- PPYOLOE+ uses Obj365 as pre-training weights, so the number of epochs for COCO training only needs 80
- YOLOX-tiny, YOLOX-s and YOLOX-m are trained with the optimizer parameters proposed in RTMDet, with different degrees of performance improvement compared to the original implementation.

Please see below items for more details

- [RTMDet](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet)
- [YOLOv5](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5)
- [YOLOv6](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov6)
- [YOLOv7](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov7)
- [YOLOv8](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov8)
- [YOLOX](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolox)
- [PPYOLO-E](https://github.com/open-mmlab/mmyolo/blob/main/configs/ppyoloe)

## VOC dataset

| Backbone | size | Batchsize | AMP | Mem (GB) | box AP(COCO metric) |
| :------: | :--: | :-------: | :-: | :------: | :-----------------: |
| YOLOv5-n | 512  |    64     | Yes |   3.5    |        51.2         |
| YOLOv5-s | 512  |    64     | Yes |   6.5    |        62.7         |
| YOLOv5-m | 512  |    64     | Yes |   12.0   |        70.1         |
| YOLOv5-l | 512  |    32     | Yes |   10.0   |        73.1         |

Please see below items for more details

- [YOLOv5](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5)

## CrowdHuman dataset

| Backbone | size | SyncBN | AMP | Mem (GB) | ignore_iof_thr | box AP50(CrowDHuman Metric) |  MR  |  JI   |
| :------: | :--: | :----: | :-: | :------: | :------------: | :-------------------------: | :--: | :---: |
| YOLOv5-s | 640  |  Yes   | Yes |   2.6    |       -1       |            85.79            | 48.7 | 75.33 |
| YOLOv5-s | 640  |  Yes   | Yes |   2.6    |      0.5       |            86.17            | 48.8 | 75.87 |

Please see below items for more details

- [YOLOv5](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5)

## DOTA 1.0 dataset
