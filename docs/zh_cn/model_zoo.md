# 模型库和评测

本页面用于汇总 MMYOLO 中支持的各类模型性能和相关评测指标，方便用户对比分析。

## COCO 数据集

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

- 所有模型均使用 COCO train2017 作为训练集，在 COCO val2017 上验证精度
- TRT-FP16-GPU-Latency(ms) 是指在 NVIDIA Tesla T4 设备上采用 TensorRT 8.4，batch size 为 1， 测试 shape 为 640x640 且仅包括模型 forward 的 GPU Compute time (YOLOX-tiny 测试 shape 是 416x416)
- 模型参数量和 FLOPs 是采用 [get_flops](https://github.com/open-mmlab/mmyolo/blob/dev/tools/analysis_tools/get_flops.py) 脚本得到，不同的运算方式可能略有不同
- RTMDet 性能是通过 [MMRazor 知识蒸馏](https://github.com/open-mmlab/mmyolo/blob/dev/configs/rtmdet/distillation/README.md) 训练后的结果
- MMYOLO 中暂时只实现了 YOLOv6 2.0 版本，并且 L 和 M 为没有经过知识蒸馏的结果
- YOLOv8 是引入了实例分割标注优化后的结果，YOLOv5、YOLOv6 和 YOLOv7 没有采用实例分割标注优化
- PPYOLOE+ 使用 Obj365 作为预训练权重，因此 COCO 训练的 epoch 数只需要 80
- YOLOX-tiny、YOLOX-s 和 YOLOX-m 为采用了 RTMDet 中所提优化器参数训练所得，性能相比原始实现有不同程度提升

详情见如下内容

- [RTMDet](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet)
- [YOLOv5](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5)
- [YOLOv6](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov6)
- [YOLOv7](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov7)
- [YOLOv8](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov8)
- [YOLOX](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolox)
- [PPYOLO-E](https://github.com/open-mmlab/mmyolo/blob/main/configs/ppyoloe)

## VOC 数据集

| Backbone | size | Batchsize | AMP | Mem (GB) | box AP(COCO metric) |
| :------: | :--: | :-------: | :-: | :------: | :-----------------: |
| YOLOv5-n | 512  |    64     | Yes |   3.5    |        51.2         |
| YOLOv5-s | 512  |    64     | Yes |   6.5    |        62.7         |
| YOLOv5-m | 512  |    64     | Yes |   12.0   |        70.1         |
| YOLOv5-l | 512  |    32     | Yes |   10.0   |        73.1         |

详情见如下内容

- [YOLOv5](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5)

## CrowdHuman 数据集

| Backbone | size | SyncBN | AMP | Mem (GB) | ignore_iof_thr | box AP50(CrowDHuman Metric) |  MR  |  JI   |
| :------: | :--: | :----: | :-: | :------: | :------------: | :-------------------------: | :--: | :---: |
| YOLOv5-s | 640  |  Yes   | Yes |   2.6    |       -1       |            85.79            | 48.7 | 75.33 |
| YOLOv5-s | 640  |  Yes   | Yes |   2.6    |      0.5       |            86.17            | 48.8 | 75.87 |

详情见如下内容

- [YOLOv5](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5)

## DOTA 1.0 数据集
