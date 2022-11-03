# 更新日志

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
