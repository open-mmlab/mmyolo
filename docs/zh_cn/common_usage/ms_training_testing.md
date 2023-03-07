# 多尺度训练和测试

## 多尺度训练

MMYOLO 中目前支持了主流的 YOLOv5、YOLOv6、YOLOv7、YOLOv8 和 RTMDet 等算法，其默认配置均为单尺度 640x640 训练。 在 MM 系列开源库中常用的多尺度训练有两种实现方式：

1. 在 `train_pipeline` 中输出的每张图都是不定尺度的，然后在 [DataPreprocessor](https://github.com/open-mmlab/mmdetection/blob/3.x/mmdet/models/data_preprocessors/data_preprocessor.py) 中将不同尺度的输入图片
   通过 [stack_batch](https://github.com/open-mmlab/mmengine/blob/dbae83c52fa54d6dda08b6692b124217fe3b2135/mmengine/model/base_model/data_preprocessor.py#L260-L261) 函数填充到同一尺度，从而组成 batch 进行训练。MMDet 中大部分算法都是采用这个实现方式。
2. 在 `train_pipeline` 中输出的每张图都是固定尺度的，然后直接在 `DataPreprocessor` 中进行 batch 张图片的上下采样，从而实现多尺度训练功能

在 MMYOLO 中两种多尺度训练方式都是支持的。理论上第一种实现方式所生成的尺度会更加丰富，但是由于其对单张图进行独立增强，训练效率不如第二种方式。所以我们更推荐使用第二种方式。

以 `configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py` 配置为例，其默认配置采用的是 640x640 固定尺度训练，假设想实现以 32 为倍数，且多尺度范围为 (480, 800) 的训练方式，则可以参考 YOLOX 做法通过 DataPreprocessor 中的 [YOLOXBatchSyncRandomResize](https://github.com/open-mmlab/mmyolo/blob/dc85144fab20a970341550794857a2f2f9b11564/mmyolo/models/data_preprocessors/data_preprocessor.py#L20) 实现。

在 `configs/yolov5` 路径下新建配置，命名为 `configs/yolov5/yolov5_s-v61_fast_1xb12-ms-40e_cat.py`，其内容如下：

```python
_base_ = 'yolov5_s-v61_fast_1xb12-40e_cat.py'

model = dict(
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='YOLOXBatchSyncRandomResize',
                # 多尺度范围是 480~800
                random_size_range=(480, 800),
                # 输出尺度需要被 32 整除
                size_divisor=32,
                # 每隔 1 个迭代改变一次输出输出
                interval=1)
        ])
)
```

上述配置就可以实现多尺度训练了。为了方便，我们已经在 `configs/yolov5/` 下已经提供了该配置。其余 YOLO 系列算法也是类似做法。

## 多尺度测试

MMYOLO 多尺度测试功能等同于测试时增强 TTA，目前已经支持，详情请查看 [测试时增强 TTA](./tta.md) 。
