# Multi-scale training and testing

## Multi-scale training

The popular YOLOv5, YOLOv6, YOLOv7, YOLOv8 and RTMDet algorithms are supported in MMYOLO currently, and their default configuration is single-scale 640x640 training. There are two implementations of multi-scale training commonly used in the MM family of open source libraries

1. Each image output in `train_pipeline` is at variable scale, and pad different scales of input images to the same scale by [stack_batch](https://github.com/open-mmlab/mmengine/blob/dbae83c52fa54d6dda08b6692b124217fe3b2135/mmengine/model/base_model/data_preprocessor.py#L260-L261) function in [DataPreprocessor](https://github.com/open-mmlab/mmdetection/blob/3.x/mmdet/models/data_preprocessors/data_preprocessor.py). Most of the algorithms in MMDet are implemented using this approach.
2. Each image output in `train_pipeline` is at a fixed scale, and `DataPreprocessor` performs up- and down-sampling of image batches for multi-scale training directly.

Both two multi-scale training approaches are supported in MMYOLO. Theoretically, the first implementation can generate richer scales, but its training efficiency is not as good as the second one due to its independent augmentation of a single image. Therefore, we recommend using the second approach.

Take `configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py` configuration as an example, its default configuration is 640x640 fixed scale training, suppose you want to implement training in multiples of 32 and multi-scale range (480, 800), you can refer to YOLOX practice by [YOLOXBatchSyncRandomResize](https://github.com/open-mmlab/mmyolo/blob/dc85144fab20a970341550794857a2f2f9b11564/mmyolo/models/data_preprocessors/data_preprocessor.py#L20) in the DataPreprocessor.

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

The above configuration will enable multi-scale training. We have already provided this configuration under `configs/yolov5/` for convenience. The rest of the YOLO family of algorithms are similar.

## Multi-scale testing

MMYOLO multi-scale testing is equivalent to Test-Time Enhancement TTA and is currently supported, see [Test-Time Augmentation TTA](./tta.md).
