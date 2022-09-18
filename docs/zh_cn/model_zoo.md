# 模型库

## 共同设置

- 所有模型都是在 `coco_2017_train` 上训练，在 `coco_2017_val` 上测试。
- 我们使用分布式训练。
- 为了与其他代码库公平比较，文档中所写的 GPU 内存是 8 个 GPU 的 `torch.cuda.max_memory_allocated()` 的最大值，此值通常小于 nvidia-smi 显示的值。
- 我们以网络 forward 和后处理的时间加和作为推理时间，不包含数据加载时间。所有结果通过 [benchmark.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/benchmark.py) 脚本计算所得。该脚本会计算推理 2000 张图像的平均时间。

## Baselines

### YOLOv5

请参考 [YOLOv5](https://github.com/open-mmlab/mmyolo/blob/master/configs/yolov5)。

### YOLOv6

请参考 [YOLOv6](https://github.com/open-mmlab/mmyolo/blob/master/configs/yolov6)。

### YOLOX

请参考 [YOLOX](https://github.com/open-mmlab/mmyolo/blob/master/configs/yolox)。
