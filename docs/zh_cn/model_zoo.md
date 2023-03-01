# 模型库和评测

本页面用于汇总 MMYOLO 中支持的各类模型性能和相关评测指标，方便用户对比分析。

## COCO 数据集

|      Model       | Arch | Size | Batch Size | Epoch | SyncBN | AMP | Mem (GB) | Params(M) | FLOPS(G) | TRT-FP16-GPU-Latency(ms) | Box AP | TTA Box AP |
| :--------------: | :--: | :--: | :--------: | :---: | :----: | :-: | :------: | :-------: | :------: | :----------------------: | :----: | :--------: |
|     YOLOv5-n     |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   1.5    |   1.87    |   2.26   |           1.14           |  28.0  |    30.7    |
|  YOLOv6-v2.0-n   |  P5  | 640  |   8xb32    |  400  |  Yes   | Yes |   6.04   |   4.32    |   5.52   |           1.37           |  36.2  |            |
|     YOLOv8-n     |  P5  | 640  |   8xb16    |  500  |  Yes   | Yes |   2.5    |   3.16    |   4.4    |           1.53           |  37.4  |    39.9    |
|   RTMDet-tiny    |  P5  | 640  |   8xb32    |  300  |  Yes   | No  |   11.7   |   4.90    |   8.09   |           2.31           |  41.8  |            |
| YOLOv6-v2.0-tiny |  P5  | 640  |   8xb32    |  400  |  Yes   | Yes |   8.13   |   9.70    |  12.37   |           2.19           |  41.0  |            |
|   YOLOv7-tiny    |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   2.7    |   6.23    |   6.89   |           1.88           |  37.5  |            |
|    YOLOX-tiny    |  P5  | 640  |   8xb32    |  300  |  Yes   | Yes |   4.9    |   5.06    |   7.63   |           1.19           |  34.3  |            |
|     RTMDet-s     |  P5  | 640  |   8xb32    |  300  |  Yes   | No  |   15.9   |   8.89    |  14.84   |           2.89           |  45.7  |            |
|     YOLOv5-s     |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   2.7    |   7.24    |   8.27   |           1.89           |  37.7  |    40.2    |
|  YOLOv6-v2.0-s   |  P5  | 640  |   8xb32    |  400  |  Yes   | Yes |   8.88   |   17.22   |  21.94   |           2.67           |  44.0  |            |
|     YOLOv8-s     |  P5  | 640  |   8xb16    |  500  |  Yes   | Yes |   4.0    |   11.17   |  14.36   |           2.61           |  45.1  |    46.8    |
|     YOLOX-s      |  P5  | 640  |   8xb32    |  300  |  Yes   | Yes |   9.8    |   8.97    |  13.40   |           2.38           |  41.9  |            |
|   PPYOLOE+ -s    |  P5  | 640  |    8xb8    |  80   |  Yes   | No  |   4.7    |   7.93    |   8.68   |           2.54           |  43.5  |            |
|     RTMDet-m     |  P5  | 640  |   8xb32    |  300  |  Yes   | No  |   27.8   |   24.71   |  39.21   |           6.23           |  50.2  |            |
|     YOLOv5-m     |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   5.0    |   21.19   |  24.53   |           4.28           |  45.3  |    46.9    |
|  YOLOv6-v2.0-m   |  P5  | 640  |   8xb32    |  300  |  Yes   | Yes |  16.69   |   34.25   |   40.7   |           5.12           |  48.4  |            |
|     YOLOv8-m     |  P5  | 640  |   8xb16    |  500  |  Yes   | Yes |   7.0    |   25.9    |  39.57   |           5.78           |  50.6  |    52.3    |
|     YOLOX-m      |  P5  | 640  |   8xb32    |  300  |  Yes   | Yes |   17.6   |   25.33   |  36.88   |           5.31           |  47.5  |            |
|   PPYOLOE+ -m    |  P5  | 640  |    8xb8    |  80   |  Yes   | No  |   8.4    |   23.43   |  24.97   |           5.47           |  49.5  |            |
|     RTMDet-l     |  P5  | 640  |   8xb32    |  300  |  Yes   | No  |   43.2   |   52.32   |  80.12   |          10.13           |  52.3  |            |
|     YOLOv5-l     |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   8.1    |   46.56   |  54.65   |           6.8            |  48.8  |    49.9    |
|  YOLOv6-v2.0-l   |  P5  | 640  |   8xb32    |  300  |  Yes   | Yes |  20.86   |   58.53   |  71.43   |           8.78           |  51.0  |            |
|     YOLOv7-l     |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   10.3   |   36.93   |  52.42   |           6.63           |  50.9  |            |
|     YOLOv8-l     |  P5  | 640  |   8xb16    |  500  |  Yes   | Yes |   9.1    |   43.69   |  82.73   |           8.97           |  53.0  |    54.4    |
|     YOLOX-l      |  P5  | 640  |    8xb8    |  300  |  Yes   | Yes |   8.0    |   54.21   |  77.83   |           9.23           |  50.1  |            |
|   PPYOLOE+ -l    |  P5  | 640  |    8xb8    |  80   |  Yes   | No  |   13.2   |   52.20   |  55.05   |           8.2            |  52.6  |            |
|     RTMDet-x     |  P5  | 640  |   8xb32    |  300  |  Yes   | No  |   63.4   |   94.86   |  145.41  |          17.89           |  52.8  |    54.2    |
|     YOLOv7-x     |  P5  | 640  |   8xb16    |  300  |  Yes   | Yes |   13.7   |   71.35   |  95.06   |          11.63           |  52.8  |            |
|     YOLOv8-x     |  P5  | 640  |   8xb16    |  500  |  Yes   | Yes |   12.4   |   68.23   |  132.10  |          14.22           |  54.0  |    55.0    |
|     YOLOX-x      |  P5  | 640  |    8xb8    |  300  |  Yes   | Yes |   9.8    |   99.07   |  144.39  |          15.35           |  51.4  |            |
|   PPYOLOE+ -x    |  P5  | 640  |    8xb8    |  80   |  Yes   | No  |   19.1   |   98.42   |  105.48  |          14.02           |  54.2  |            |

- TRT-FP16-GPU-Latency(ms) 是指在 NVIDIA tesla T4 设备上采用 TensorRT 8.4，bs 为 1， 测试 shape 为 640x640 且不包括后处理的 GPU Compute time
- 模型参数量和 FLOPS 是采用 [get_flops](https://github.com/open-mmlab/mmyolo/blob/dev/tools/analysis_tools/get_flops.py) 脚本得到，不同的运算方式可能略有不同
- RTMDet 性能是通过 MMRazor 知识蒸馏的
- MMYOLO 中暂时只实现了 YOLOv6 2.0 版本，并且 L 和 M 为没有经过知识蒸馏的结果
- YOLOv8 是引入了实例分割标注优化后的结果，YOLOv5、YOLOv6 和 YOLOv7 没有采用实例分割标注优化
- PPYOLOE+ 使用 Obj365 作为预训练权重，因此 COCO 训练的 epoch 数只需要 80
- YOLOX-tiny、YOLOX-s 和 YOLOX-m 为采用了 RTMDet 中所提优化器参数训练所得，性能相比原始实现有不同程度提升

### 详情

- [RTMDet](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet)
- [YOLOv5](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5)
- [YOLOv6](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov6)
- [YOLOv7](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov7)
- [YOLOv8](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov8)
- [YOLOX](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolox)
- [PPYOLO-E](https://github.com/open-mmlab/mmyolo/blob/main/configs/ppyoloe)

## VOC 数据集

## CrowdHuman 数据集

## DOTA 1.0 数据集
