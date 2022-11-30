# 更新日志

## v0.2.0（1/12/2022)

### 亮点

1. 支持 [YOLOv7](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov7) P5 和 P6 模型
2. 支持 [YOLOv6](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov6/README.md) 中的 ML 大模型
3. 支持 [Grad-Based CAM 和 Grad-Free CAM](https://github.com/open-mmlab/mmyolo/blob/dev/demo/boxam_vis_demo.py)
4. 基于 sahi 支持[大图推理](https://github.com/open-mmlab/mmyolo/blob/dev/demo/large_image_demo.py)
5. projects 文件夹下新增 [easydeploy](https://github.com/open-mmlab/mmyolo/blob/dev/projects/easydeploy/README.md) 项目
6. 新增 [自定义数据集教程](https://github.com/open-mmlab/mmyolo/blob/dev/docs/zh_cn/user_guides/custom_dataset.md)

### 新特性

1. `browse_dataset.py` 脚本支持可视化原图、数据增强后和中间结果功能 (#304)
2. `image_demo.py` 新增预测结果保存为 labelme 格式功能 (#288, #314)
3. 新增 labelme 格式转 COCO 格式脚本 `labelme2coco` (#308, #313)
4. 新增 COCO 数据集切分脚本 `coco_split.py` (#311)
5. `how-to.md` 文档中新增两个 backbone 替换案例以及更新 `plugin.md` (#291)
6. 新增贡献者文档 `contributing.md` and 代码规范文档 `code_style.md` (#322)
7. 新增如何通过 mim 跨库调用脚本文档 (#321)
8. `YOLOv5` 支持 RV1126 设备部署 (#262)

### Bug 修复

1. 修复 `MixUp` padding 错误 (#319)
2. 修复 `LetterResize` 和 `YOLOv5KeepRatioResize` 中 `scale_factor` 参数顺序错误 (#305)
3. 修复 `YOLOX Nano` 模型训练错误问题 (#285)
4. 修复 `RTMDet` 部署没有导包的错误 (#287)
5. 修复 int8 部署配置错误 (#315)
6. 修复 `basebackbone` 中 `make_stage_plugins` 注释 (#296)
7. 部署模块支持切换为 deploy 模式功能 (#324)
8. 修正 `RTMDet` 模型结构图中的错误 (#317)

### 完善

1. `test.py` 中新增 json 格式导出选项 (#316)
2. `extract_subcoco.py` 脚本中新增基于面积阈值过滤规则 (#286)
3. 部署相关中文文档翻译为英文 (#289)
4. 新增 `YOLOv6` 算法描述大纲文档 (#252)
5. 完善 `config.md` (#297, #303)
6. 完善 `mosiac9` 的 docstring (#307)
7. 完善 `browse_coco_json.py` 脚本输入参数 (#309)
8. 重构 `dataset_analysis.py` 中部分函数使其更加通用 (#294)

### 视频

1. 发布了 [工程文件结构简析](https://www.bilibili.com/video/BV1LP4y117jS)
2. 发布了 [10分钟换遍主干网络文档](https://www.bilibili.com/video/BV1JG4y1d7GC)

### 贡献者

总共 14 位开发者参与了本次版本

谢谢 @fcakyon, @matrixgame2018, @MambaWong, @imAzhou, @triple-Mu, @RangeKing, @PeterH0323, @xin-li-67, @kitecats, @hanrui1sensetime, @AllentDan, @Zheng-LinXiao, @hhaAndroid, @wanghonglie

## v0.1.3（10/11/2022)

### 新特性

1. 支持 CBAM 插件并提供插件文档 (#246)
2. 新增 YOLOv5 P6 模型结构图和相关说明 (#273)

### Bug 修复

1. 基于 mmengine 0.3.1 修复保存最好权重时训练失败问题
2. 基于 mmdet 3.0.0rc3 修复 `add_dump_metric` 报错 (#253)
3. 修复 backbone 不支持 `init_cfg` 问题 (#272)
4. 基于 mmdet 3.0.0rc3 改变 typing 导入方式 (#261)

### 完善

1. `featmap_vis_demo` 支持文件夹和 url 输入 (#248)
2. 部署 docker 文件完善 (#242)

### 贡献者

总共 10 位开发者参与了本次版本

谢谢 @kitecats, @triple-Mu, @RangeKing, @PeterH0323, @Zheng-LinXiao, @tkhe, @weikai520, @zytx121, @wanghonglie, @hhaAndroid

## v0.1.2（3/11/2022)

### 亮点

1. 支持 ONNXRuntime 和 TensorRT 的 [YOLOv5/YOLOv6/YOLOX/RTMDet 部署](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy)
2. 支持 [YOLOv6](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov6) s/t/n 模型训练
3. YOLOv5 支持 [P6 大分辨率 1280 尺度训练](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5)
4. YOLOv5 支持 [VOC 数据集训练](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/voc)
5. 支持 [PPYOLOE](https://github.com/open-mmlab/mmyolo/blob/main/configs/ppyoloe) 和 [YOLOv7](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov7) 模型推理和官方权重转化
6. How-to 文档中新增 YOLOv5 替换 [backbone 教程](https://github.com/open-mmlab/mmyolo/blob/dev/docs/zh_cn/advanced_guides/how_to.md#%E8%B7%A8%E5%BA%93%E4%BD%BF%E7%94%A8%E4%B8%BB%E5%B9%B2%E7%BD%91%E7%BB%9C)

### 新特性

1. 新增 `optimize_anchors` 脚本 (#175)
2. 新增 `extract_subcoco` 脚本 (#186)
3. 新增 `yolo2coco` 转换脚本 (#161)
4. 新增 `dataset_analysis` 脚本 (#172)
5. 移除 Albu 版本限制 (#187)

### Bug 修复

1. 修复当设置 `cfg.resume` 时候不生效问题 (#221)
2. 修复特征图可视化脚本中不显示 bbox 问题 (#204)
3. 更新 RTMDet 的 metafile (#188)
4. 修复 test_pipeline 中的可视化错误 (#166)
5. 更新 badges (#140)

### 完善

1. 优化 Readthedoc 显示页面 (#209)
2. 为 base model 添加模块结构图的 docstring (#196)
3. 支持 LoadAnnotations 中不包括任何实例逻辑 (#161)
4. 更新 `image_demo` 脚本以支持文件夹和 url 路径 (#128)
5. 更新 pre-commit hook (#129)

### 文档

1. 将 `yolov5_description.md`、 `yolov5_tutorial.md` 和 `visualization.md` 翻译为英文 (#138, #198, #206)
2. 新增部署相关中文文档 (#220)
3. 更新 `config.md`、`faq.md` 和 `pull_request_template.md` (#190, #191, #200)
4. 更新 `article` 页面 (#133)

### 视频

1. 发布了[特征图可视化视频](https://www.bilibili.com/video/BV188411s7o8)
2. 发布了 [YOLOv5 配置文件解读视频](https://www.bilibili.com/video/BV1214y157ck)
3. 发布了 [RTMDet-s 特征图可视化 demo 视频](https://www.bilibili.com/video/BV1je4y1478R)
4. 发布了[源码解读和必备调试技巧视频](https://www.bilibili.com/video/BV1N14y1V7mB)

### 贡献者

总共 14 位开发者参与了本次版本

谢谢 @imAzhou, @triple-Mu, @RangeKing, @PeterH0323, @xin-li-67, @Nioolek, @kitecats, @Bin-ze, @JiayuXu0, @cydiachen, @zhiqwang, @Zheng-LinXiao, @hhaAndroid, @wanghonglie

## v0.1.1（29/9/2022)

基于 MMDetection 的 RTMDet 高精度低延时目标检测算法，我们也同步发布了 RTMDet，并提供了 RTMDet 原理和实现全解析中文文档

### 亮点

1. 支持了 [RTMDet](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet)
2. 新增了 [RTMDet 原理和实现全解析中文文档](https://github.com/open-mmlab/mmyolo/blob/main/docs/zh_cn/algorithm_descriptions/rtmdet_description.md)
3. 支持对 backbone 自定义插件，并更新了 How-to 文档 (#75)

### Bug 修复

1. 修复一些文档错误 (#66, #72, #76, #83, #86)
2. 修复权重链接错误 (#63)
3. 修复 `LetterResize` 使用 `imscale` api 时候输出不符合预期的 bug (#105)

### 完善

1. 缩减 docker 镜像尺寸 (#67)
2. 简化 BaseMixImageTransform 中 Compose 逻辑 (#71)
3. test 脚本支持 dump 结果 (#84)

#### 贡献者

总共 13 位开发者参与了本次版本

谢谢 @wanghonglie, @hhaAndroid, @yang-0201, @PeterH0323, @RangeKing, @satuoqaq, @Zheng-LinXiao, @xin-li-67, @suibe-qingtian, @MambaWong, @MichaelCai0912, @rimoire, @Nioolek

## v0.1.0（21/9/2022)

我们发布了 MMYOLO 开源库，其基于 MMEngine, MMCV 2.x 和 MMDetection 3.x 库. 目前实现了目标检测功能，后续会扩展为多任务。

### 亮点

1. 支持 YOLOv5/YOLOX 训练，支持 YOLOv6 推理。部署即将支持。
2. 重构了 MMDetection 的 YOLOX，提供了更快的训练和推理速度。
3. 提供了详细入门和进阶教程, 包括 YOLOv5 从入门到部署、YOLOv5 算法原理和实现全解析、 特征图可视化等教程。
