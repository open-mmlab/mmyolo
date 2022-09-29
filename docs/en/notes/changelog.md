# Changelog

## v0.1.1（29/9/2022)

Based on MMDetection's RTMDet high precision and low latency object detection algorithm, we have also released RTMDet and provided a Chinese document on the principle and implementation of RTMDet.

### Highlights

1. Support [RTMDet](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet)
2. Support for backbone customization plugins and update How-to documentation (#75)

### Bug Fixes

1. Fix some documentation errors (#66, #72, #76, #83, #86)
2. Fix checkpoints link error (#63)

### Improvements

1. Reducing the size of docker images (#67)
2. Simplifying `Compose` Logic in `BaseMixImageTransform` (#71)
3. Supports dump results in `test.py` (#84)

#### Contributors

A total of 12 developers contributed to this release.

Thank @wanghonglie, @hhaAndroid, @yang-0201, @PeterH0323, @RangeKing, @satuoqaq, @Zheng-LinXiao, @xin-li-67, @suibe-qingtian, @MambaWong, @MichaelCai0912, @rimoire

## v0.1.0（21/9/2022)

We have released MMYOLO open source library, which is based on MMEngine, MMCV 2.x and MMDetection 3.x libraries. At present, the object detection has been realized, and it will be expanded to multi-task in the future.

### Highlights

1. Support YOLOv5/YOLOX training, support YOLOv6 inference. Deployment will be supported soon.
2. Refactored YOLOX from MMDetection to accelerate training and inference.
3. Detailed introduction and advanced tutorials are provided, see the [English tutorial](https://mmyolo.readthedocs.io/en/latest).
