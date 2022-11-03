# Changelog

## v0.1.2（3/11/2022)

### Highlights

1. Support [YOLOv5/YOLOv6/YOLOX/RTMDet deployments](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy) for ONNXRuntime and TensorRT
2. Support [YOLOv6](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov6) s/t/n model training
3. YOLOv5 supports [P6 model training which can input 1280-scale images](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5)
4. YOLOv5 supports [VOC dataset training](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/voc)
5. Support [PPYOLOE](https://github.com/open-mmlab/mmyolo/blob/main/configs/ppyoloe) and [YOLOv7](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov7) model inference and official weight conversion
6. Add YOLOv5 replacement [backbone tutorial](https://github.com/open-mmlab/mmyolo/blob/dev/docs/en/advanced_guides/how_to.md#use-backbone-network-implemented-in-other-openmmlab-repositories) in How-to documentation

### New Features

1. Add `optimize_anchors` script (#175)
2. Add `extract_subcoco` script (#186)
3. Add `yolo2coco` conversion script (#161)
4. Add `dataset_analysis` script (#172)
5. Remove Albu version restrictions (#187)

### Bug Fixes

1. Fix the problem that `cfg.resume` does not work when set (#221)
2. Fix the problem of not showing bbox in feature map visualization script (#204)
3. uUpdate the metafile of RTMDet (#188)
4. Fix a visualization error in `test_pipeline` (#166)
5. Update badges (#140)

### Improvements

1. Optimize Readthedoc display page (#209)
2. Add docstring for module structure diagram for base model (#196)
3. Support for not including any instance logic in LoadAnnotations (#161)
4. Update `image_demo` script to support folder and url paths (#128)
5. Update pre-commit hook (#129)

### Documentation

1. Translate `yolov5_description.md`, `yolov5_tutorial.md` and `visualization.md` into English (#138, #198, #206)
2. Add deployment-related Chinese documentation (#220)
3. Update `config.md`, `faq.md` and `pull_request_template.md` (#190, #191, #200)
4. Update the `article` page (#133)

#### Contributors

A total of 14 developers contributed to this release.

Thank @imAzhou, @triple-Mu, @RangeKing, @PeterH0323, @xin-li-67, @Nioolek, @kitecats, @Bin-ze, @JiayuXu0, @cydiachen, @zhiqwang, @Zheng-LinXiao, @hhaAndroid, @wanghonglie

## v0.1.1（29/9/2022)

Based on MMDetection's RTMDet high precision and low latency object detection algorithm, we have also released RTMDet and provided a Chinese document on the principle and implementation of RTMDet.

### Highlights

1. Support [RTMDet](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet)
2. Support for backbone customization plugins and update How-to documentation (#75)

### Bug Fixes

1. Fix some documentation errors (#66, #72, #76, #83, #86)
2. Fix checkpoints link error (#63)
3. Fix the bug that the output of `LetterResize` does not meet the expectation when using `imscale` (#105)

### Improvements

1. Reducing the size of docker images (#67)
2. Simplifying `Compose` Logic in `BaseMixImageTransform` (#71)
3. Supports dump results in `test.py` (#84)

#### Contributors

A total of 13 developers contributed to this release.

Thank @wanghonglie, @hhaAndroid, @yang-0201, @PeterH0323, @RangeKing, @satuoqaq, @Zheng-LinXiao, @xin-li-67, @suibe-qingtian, @MambaWong, @MichaelCai0912, @rimoire, @Nioolek

## v0.1.0（21/9/2022)

We have released MMYOLO open source library, which is based on MMEngine, MMCV 2.x and MMDetection 3.x libraries. At present, the object detection has been realized, and it will be expanded to multi-task in the future.

### Highlights

1. Support YOLOv5/YOLOX training, support YOLOv6 inference. Deployment will be supported soon.
2. Refactored YOLOX from MMDetection to accelerate training and inference.
3. Detailed introduction and advanced tutorials are provided, see the [English tutorial](https://mmyolo.readthedocs.io/en/latest).
