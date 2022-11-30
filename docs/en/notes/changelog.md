# Changelog

## v0.2.0（1/12/2022)

### Highlights

1. Support [YOLOv7](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov7) P5 and P6 model
2. Support [YOLOv6](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov6/README.md) ML model
3. Support [Grad-Based CAM and Grad-Free CAM](https://github.com/open-mmlab/mmyolo/blob/dev/demo/boxam_vis_demo.py)
4. Support [large image inference](https://github.com/open-mmlab/mmyolo/blob/dev/demo/large_image_demo.py) based on sahi
5. Add [easydeploy](https://github.com/open-mmlab/mmyolo/blob/dev/projects/easydeploy/README.md) project under the projects folder
6. Add [custom dataset guide](https://github.com/open-mmlab/mmyolo/blob/dev/docs/zh_cn/user_guides/custom_dataset.md)

### New Features

1. `browse_dataset.py` script supports visualization of original image, data augmentation and intermediate results (#304)
2. Add flag to output labelme label file in `image_demo.py` (#288, #314)
3. Add `labelme2coco` script (#308, #313)
4. Add split COCO dataset script (#311)
5. Add two examples of backbone replacement in `how-to.md` and update `plugin.md` (#291)
6. Add `contributing.md` and `code_style.md` (#322)
7. Add docs about how to use mim to run scripts across libraries (#321)
8. Support YOLOv5 deployment at RV1126 device (#262)

### Bug Fixes

1. Fix MixUp padding error (#319)
2. Fix scale factor order error of `LetterResize` and `YOLOv5KeepRatioResize` (#305)
3. Fix training errors of `YOLOX Nano` model (#285)
4. Fix `RTMDet` deploy error (#287)
5. Fix int8 deploy config (#315)
6. Fix `make_stage_plugins` doc in `basebackbone` (#296)
7. Enable switch to deploy when create pytorch model in deployment (#324)

### Improvements

1. Add option of json output in `test.py` (#316)
2. Add area condition in `extract_subcoco.py` script (#286)
3. Update `RTMDet` model graph (#317)
4. Deployment doc translation (#289)
5. Add YOLOv6 description overview doc (#252)
6. Improve `config.md` (#297, #303)
7. Add mosaic9 graph in docstring  (#307)
8. Improve `browse_coco_json.py` script args (#309)
9. Refactor some functions in `dataset_analysis.py` to be more general (#294)

#### Contributors

A total of 14 developers contributed to this release.

Thank  @fcakyon, @matrixgame2018, @MambaWong, @imAzhou, @triple-Mu, @RangeKing, @PeterH0323, @xin-li-67, @kitecats, @hanrui1sensetime, @AllentDan, @Zheng-LinXiao, @hhaAndroid, @wanghonglie

## v0.1.3（10/11/2022)

### New Features

1. Support CBAM plug-in and provide plug-in documentation (#246)
2. Add YOLOv5 P6 model structure diagram and related descriptions (#273)

### Bug Fixes

1. Fix training failure when saving best weights based on mmengine 0.3.1
2. Fix `add_dump_metric` error based on mmdet 3.0.0rc3 (#253)
3. Fix backbone does not support `init_cfg` issue (#272)
4. Change typing import method based on mmdet 3.0.0rc3 (#261)

### Improvements

1. `featmap_vis_demo` support for folder and url input (#248)
2. Deploy docker file refinement (#242)

#### Contributors

A total of 10 developers contributed to this release.

Thank @kitecats, @triple-Mu, @RangeKing, @PeterH0323, @Zheng-LinXiao, @tkhe, @weikai520, @zytx121, @wanghonglie, @hhaAndroid

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
