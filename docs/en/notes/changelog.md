# Changelog

## v0.5.0 (2/3/2023)

### Highlights

1. Support [RTMDet-R](https://github.com/open-mmlab/mmyolo/blob/dev/configs/rtmdet/README.md#rotated-object-detection) rotated object detection
2. Support for using mask annotation to improve [YOLOv8](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov8/README.md) object detection performance
3. Support [MMRazor](https://github.com/open-mmlab/mmyolo/blob/dev/configs/razor/subnets/README.md) searchable NAS sub-network as the backbone of YOLO series algorithm
4. Support calling [MMRazor](https://github.com/open-mmlab/mmyolo/blob/dev/configs/rtmdet/distillation/README.md) to distill the knowledge of RTMDet
5. [MMYOLO](https://mmyolo.readthedocs.io/zh_CN/dev/) document structure optimization, comprehensive content upgrade
6. Improve YOLOX mAP and training speed based on RTMDet training hyperparameters
7. Support calculation of model parameters and FLOPs, provide GPU latency data on T4 devices, and update [Model Zoo](https://github.com/open-mmlab/mmyolo/blob/dev/docs/en/model_zoo.md)
8. Support test-time augmentation (TTA)
9. Support RTMDet, YOLOv8 and YOLOv7 assigner visualization

### New Features

01. Support inference for RTMDet instance segmentation tasks (#583)
02. Beautify the configuration file in MMYOLO and add more comments (#501, #506, #516, #529, #531, #539)
03. Refactor and optimize documentation (#568, #573, #579, #584, #587, #589, #596, #599, #600)
04. Support fast version of YOLOX (#518)
05. Support DeepStream in EasyDeploy and add documentation (#485, #545, #571)
06. Add confusion matrix drawing script (#572)
07. Add single channel application case (#460)
08. Support auto registration (#597)
09. Support Box CAM of YOLOv7, YOLOv8 and PPYOLOE (#601)
10. Add automated generation of MM series repo registration information and tools scripts (#559)
11. Added YOLOv7 model structure diagram (#504)
12. Add how to specify specific GPU training and inference files (#503)
13. Add check if `metainfo` is all lowercase when training or testing (#535)
14. Add links to Twitter, Discord, Medium, YouTube, etc. (#555)

### Bug Fixes

1. Fix isort version issue (#492, #497)
2. Fix type error of assigner visualization (#509)
3. Fix YOLOv8 documentation link error (#517)
4. Fix RTMDet Decoder error in EasyDeploy (#519)
5. Fix some document linking errors (#537)
6. Fix RTMDet-Tiny weight path error (#580)

### Improvements

1. Update `contributing.md`
2. Optimize `DetDataPreprocessor` branch to support multitasking (#511)
3. Optimize `gt_instances_preprocess` so it can be used for other YOLO algorithms (#532)
4. Add `yolov7-e6e` weight conversion script (#570)
5. Reference YOLOv8 inference code modification PPYOLOE

### Contributors

A total of 22 developers contributed to this release.

Thank @triple-Mu, @isLinXu, @Audrey528, @TianWen580, @yechenzhi, @RangeKing, @lyviva, @Nioolek, @PeterH0323, @tianleiSHI, @aptsunny, @satuoqaq, @vansin, @xin-li-67, @VoyagerXvoyagerx,
@landhill, @kitecats, @tang576225574, @HIT-cwh, @AI-Tianlong, @RangiLyu, @hhaAndroid

## v0.4.0 (18/1/2023)

### Highlights

1. Implemented [YOLOv8](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov8/README.md) object detection model, and supports model deployment in [projects/easydeploy](https://github.com/open-mmlab/mmyolo/blob/dev/projects/easydeploy)
2. Added Chinese and English versions of [Algorithm principles and implementation with YOLOv8](https://github.com/open-mmlab/mmyolo/blob/dev/docs/en/algorithm_descriptions/yolov8_description.md)

### New Features

1. Added YOLOv8 and PPYOLOE model structure diagrams (#459, #471)
2. Adjust the minimum supported Python version from 3.6 to 3.7 (#449)
3. Added a new YOLOX decoder in TensorRT-8 (#450)
4. Add a tool for scheduler visualization (#479)

### Bug Fixes

1. Fix `optimize_anchors.py` script import error (#452)
2. Fix the wrong installation steps in `get_started.md` (#474)
3. Fix the neck error when using the `RTMDet` P6 model (#480)

### Contributors

A total of 9 developers contributed to this release.

Thank @VoyagerXvoyagerx, @tianleiSHI, @RangeKing, @PeterH0323, @Nioolek, @triple-Mu, @lyviva, @Zheng-LinXiao, @hhaAndroid

## v0.3.0 (8/1/2023)

### Highlights

1. Implement fast version of [RTMDet](https://github.com/open-mmlab/mmyolo/blob/dev/configs/rtmdet/README.md). RTMDet-s 8xA100 training takes only 14 hours. The training speed is 2.6 times faster than the previous version.
2. Support [PPYOLOE](https://github.com/open-mmlab/mmyolo/blob/dev/configs/ppyoloe/README.md) training
3. Support `iscrowd` attribute training in [YOLOv5](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov5/crowdhuman/yolov5_s-v61_8xb16-300e_ignore_crowdhuman.py)
4. Support [YOLOv5 assigner result visualization](https://github.com/open-mmlab/mmyolo/blob/dev/projects/assigner_visualization/README.md)

### New Features

01. Add `crowdhuman` dataset (#368)
02. Easydeploy support TensorRT inference (#377)
03. Add `YOLOX` structure description (#402)
04. Add a feature for the video demo (#392)
05. Support `YOLOv7` easy deploy (#427)
06. Add resume from specific checkpoint in CLI (#393)
07. Set `metainfo` fields to lower case (#362, #412)
08. Add module combination doc (#349, #352, #345)
09. Add docs about how to freeze the weight of backbone or neck (#418)
10. Add don't used pre-training weights doc in `how_to.md` (#404)
11. Add docs about how to set the random seed (#386)
12. Translate `rtmdet_description.md` document to English (#353)
13. Add doc of `yolov6_description.md` (#382, #372)

### Bug Fixes

01. Fix bugs in the output annotation file when `--class-id-txt` is set (#430)
02. Fix batch inference bug in `YOLOv5` head (#413)
03. Fix typehint in some heads (#415, #416, #443)
04. Fix RuntimeError of `torch.cat()` expected a non-empty list of Tensors (#376)
05. Fix the device inconsistency error in `YOLOv7` training (#397)
06. Fix the `scale_factor` and `pad_param` value in `LetterResize` (#387)
07. Fix docstring graph rendering error of readthedocs (#400)
08. Fix AssertionError when `YOLOv6` from training to val (#378)
09. Fix CI error due to `np.int` and legacy builder.py (#389)
10. Fix MMDeploy rewriter (#366)
11. Fix MMYOLO unittest scope bug (#351)
12. Fix `pad_param` error (#354)
13. Fix twice head inference bug (#342)
14. Fix customize dataset training (#428)

### Improvements

01. Update `useful_tools.md` (#384)
02. update the English version of `custom_dataset.md` (#381)
03. Remove context argument from the rewriter function (#395)
04. deprecating `np.bool` type alias (#396)
05. Add new video link for custom dataset (#365)
06. Export onnx for model only (#361)
07. Add MMYOLO regression test yml (#359)
08. Update video tutorials in `article.md` (#350)
09. Add deploy demo (#343)
10. Optimize the vis results of large images in debug mode (#346)
11. Improve args for `browse_dataset` and support `RepeatDataset` (#340, #338)

### Contributors

A total of 28 developers contributed to this release.

Thank @RangeKing, @PeterH0323, @Nioolek, @triple-Mu, @matrixgame2018, @xin-li-67, @tang576225574, @kitecats, @Seperendity, @diplomatist, @vaew, @wzr-skn, @VoyagerXvoyagerx, @MambaWong, @tianleiSHI, @caj-github, @zhubochao, @lvhan028, @dsghaonan, @lyviva, @yuewangg, @wang-tf, @satuoqaq, @grimoire, @RunningLeon, @hanrui1sensetime, @RangiLyu, @hhaAndroid

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
8. Support `YOLOv5` deployment at RV1126 device (#262)

### Bug Fixes

1. Fix MixUp padding error (#319)
2. Fix scale factor order error of `LetterResize` and `YOLOv5KeepRatioResize` (#305)
3. Fix training errors of `YOLOX Nano` model (#285)
4. Fix `RTMDet` deploy error (#287)
5. Fix int8 deploy config (#315)
6. Fix `make_stage_plugins` doc in `basebackbone` (#296)
7. Enable switch to deploy when create pytorch model in deployment (#324)
8. Fix some errors in `RTMDet` model graph (#317)

### Improvements

1. Add option of json output in `test.py` (#316)
2. Add area condition in `extract_subcoco.py` script (#286)
3. Deployment doc translation (#289)
4. Add YOLOv6 description overview doc (#252)
5. Improve `config.md` (#297, #303)
   6Add mosaic9 graph in docstring  (#307)
6. Improve `browse_coco_json.py` script args (#309)
7. Refactor some functions in `dataset_analysis.py` to be more general (#294)

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
