# 概述

## MMYOLO 介绍

<div align=center>
<img src="https://user-images.githubusercontent.com/45811724/190993591-bd3f1f11-1c30-4b93-b5f4-05c9ff64ff7f.gif" alt="图片"/>
</div>

MMYOLO 是一个基于 PyTorch 和 MMDetection 的 YOLO 系列算法开源工具箱，它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。 MMYOLO 定位为 YOLO 系列热门开源库以及工业应用核心库，其愿景图如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/220060451-d50414e5-a239-45b7-a4db-ed8699820300.png" alt="愿景图"/>
</div>

目前支持的任务如下：

<details open>
<summary><b>支持的任务</b></summary>

- 目标检测
- 旋转框目标检测

</details>

目前支持的 YOLO 系列算法如下：

<details open>
<summary><b>支持的算法</b></summary>

- YOLOv5
- YOLOX
- RTMDet
- RTMDet-Rotated
- YOLOv6
- YOLOv7
- PPYOLOE
- YOLOv8

</details>

目前支持的数据集如下：

<details open>
<summary><b>支持的数据集</b></summary>

- COCO Dataset
- VOC Dataset
- CrowdHuman Dataset
- DOTA 1.0 Dataset

</details>

MMYOLO 支持在 Linux、Windows、macOS 上运行， 支持 PyTorch 1.7 及其以上版本运行。它具有如下三个特性：

- 🕹️ **统一便捷的算法评测**

  MMYOLO 统一了各类 YOLO 算法模块的实现，并提供了统一的评测流程，用户可以公平便捷地进行对比分析。

- 📚 **丰富的入门和进阶文档**

  MMYOLO 提供了从入门到部署到进阶和算法解析等一系列文档，方便不同用户快速上手和扩展。

- 🧩 **模块化设计**

  MMYOLO 将框架解耦成不同的模块组件，通过组合不同的模块和训练测试策略，用户可以便捷地构建自定义模型。

<img src="https://user-images.githubusercontent.com/27466624/199999337-0544a4cb-3cbd-4f3e-be26-bcd9e74db7ff.jpg" alt="基类-P5"/>
  图为 RangeKing@GitHub 提供，非常感谢！

## 本文档使用指南

MMYOLO 中将文档结构分成 6 个部分，对应不同需求的用户。

- **开启 MMYOLO 之旅**。本部分是第一次使用 MMYOLO 用户的必读文档，请全文仔细阅读
- **推荐专题**。本部分是 MMYOLO 中提供的以主题形式的精华文档，包括了 MMYOLO 中大量的特性等。强烈推荐使用 MMYOLO 的所有用户阅读
- **常用功能**。本部分提供了训练测试过程中用户经常会用到的各类常用功能，用户可以在用到时候再次查阅
- **实用工具**。本部分是 tools 下使用工具的汇总文档，便于大家能够快速的愉快使用 MMYOLO 中提供的各类脚本
- **基础和进阶教程**。本部分涉及到 MMYOLO 中的一些基本概念和进阶教程等，适合想详细了解 MMYOLO 设计思想和结构设计的用户
- **其他**。其余部分包括模型仓库、说明和接口文档等等

不同需求的用户可以按需选择你心怡的内容阅读。如果你对本文档有异议或者更好的优化办法，欢迎给 MMYOLO 提 PR ～, 请参考 [如何给 MMYOLO 贡献代码](../recommended_topics/contributing.md)
