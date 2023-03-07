# Multi-scale training and testing

## Multi-scale training

The popular YOLOv5, YOLOv6, YOLOv7, YOLOv8 and RTMDet algorithms are supported in MMYOLO currently, and their default configuration is single-scale 640x640 training. There are two implementations of multiscale training commonly used in the MM family of open source libraries

1. Each image output in `train_pipeline` is at variable scale, and pad different scales of input images to the same scale by [stack_batch](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/data_preprocessor.py#L260) function in [DataPreprocessor](https://github.com/open-mmlab/mmdetection/blob/3.x/mmdet/models/data_preprocessors/data_preprocessor.py). Most of the algorithms in MMDet are implemented using this approach.
2. Each image output in `train_pipeline` is at fixed scale, and then up and down sampling of batches of images directly in `DataPreprocessor` for multi-scale training.

Theoretically, the first implementation generates richer scales, but due to its independent data augmentation of a single image, the training efficiency is not as efficient as the second one. Both multi-scale training methods are supported in MMYOLO, but I recommend you use the second one.

Take `configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py` configuration as an example, its default configuration is 640x640 fixed scale training, suppose you want to implement training in multiples of 32 and multi-scale range (480, 800), you can refer to YOLOX practice by [YOLOXBatchSyncRandomResize](https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/models/data_preprocessors/data_preprocessor.py#L20) in the DataPreprocessor.

Create a new configuration under the `configs/yolov5` path named `configs/yolov5/yolov5_s-v61_fast_1xb12-ms-40e_cat.py` with the following contents.

```python
_base_ = 'yolov5_s-v61_fast_1xb12-40e_cat.py'

model = dict(
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='YOLOXBatchSyncRandomResize',
                # multi-scale range (480, 800)
                random_size_range=(480, 800),
                # The output scale needs to be divisible by 32
                size_divisor=32,
                interval=1)
        ])
)
```

The above configuration will enable multi-scale training. We have already provided this configuration under `configs/yolov5/` for convenience. The rest of the YOLO family of algorithms is similar.

## Multi-scale testing

MMYOLO multi-scale testing is equivalent to Test-Time Enhancement TTA and is currently supported, see [Test-Time Augmentation TTA](./tta.md).
