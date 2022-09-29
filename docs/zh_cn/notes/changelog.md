# 更新日志

## v0.1.1（29/9/2022)

基于 MMDetection 的 RTMDet 高精度低延时目标检测算法，我们也同步发布了 RTMDet，并提供了 RTMDet 原理和实现全解析中文文档

### 亮点

1. 支持了 [RTMDet](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet)
2. 新增了 [RTMDet 原理和实现全解析中文文档](https://github.com/open-mmlab/mmyolo/blob/main/docs/zh_cn/algorithm_descriptions/rtmdet_description.md)
3. 支持对 backbone 自定义插件，并更新了 How-to 文档 (#75)

### Bug 修复

1. 修复一些文档错误 (#66, #72, #76, #83, #86)
2. 修复权重链接错误 (#63)

### 提高

1. 缩减 docker 镜像尺寸 (#67)
2. 简化 BaseMixImageTransform 中 Compose 逻辑 (#71)
3. test 脚本支持 dump 结果 (#84)

#### 贡献者

总共 12 位开发者参与了本次版本

谢谢 @wanghonglie, @hhaAndroid, @yang-0201, @PeterH0323, @RangeKing, @satuoqaq, @Zheng-LinXiao, @xin-li-67, @suibe-qingtian, @MambaWong, @MichaelCai0912, @rimoire

## v0.1.0（21/9/2022)

我们发布了 MMYOLO 开源库，其基于 MMEngine, MMCV 2.x 和 MMDetection 3.x 库. 目前实现了目标检测功能，后续会扩展为多任务。

### 亮点

1. 支持 YOLOv5/YOLOX 训练，支持 YOLOv6 推理。部署即将支持。
2. 重构了 MMDetection 的 YOLOX，提供了更快的训练和推理速度。
3. 提供了详细入门和进阶教程, 包括 YOLOv5 从入门到部署、YOLOv5 算法原理和实现全解析、 特征图可视化等教程。
