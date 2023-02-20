# 概述

本章向您介绍 MMYOLO 的整体框架，并提供详细的教程链接。

## MMYOLO 介绍

<div align=center>
<img src="https://user-images.githubusercontent.com/45811724/190993591-bd3f1f11-1c30-4b93-b5f4-05c9ff64ff7f.gif" alt="图片"/>
</div>

MMYOLO 是一个 YOLO 系列的算法工具箱，目前仅实现了目标检测任务，后续会支持实例分割、全景分割和关键点检测等多种任务。其包括丰富的目标检测算法以及相关的组件和模块，下面是它的整体框架：

MMYOLO 文件结构和 MMDetection 完全一致。为了能够充分复用 MMDetection 代码，MMYOLO 仅包括定制内容，其由 3 个主要部分组成：`datasets`、`models`、`engine`。

- **datasets** 支持用于目标检测的各种数据集。
  - **transforms** 包含各种数据增强变换。
- **models** 是检测器最重要的部分，包含检测器的不同组件。
  - **detectors** 定义所有检测模型类。
  - **data_preprocessors** 用于预处理模型的输入数据。
  - **backbones** 包含各种骨干网络
  - **necks** 包含各种模型颈部组件
  - **dense_heads** 包含执行密集预测的各种检测头。
  - **losses** 包含各种损失函数
  - **task_modules** 为检测任务提供模块。例如 assigners、samplers、box coders 和 prior generators。
  - **layers** 提供了一些基本的神经网络层
- **engine** 是运行时组件的一部分。
  - **optimizers** 提供优化器和优化器封装。
  - **hooks** 提供 runner 的各种钩子。

## 文档使用指南

MMYOLO 中将文档结构分成 6 个部分，对应不同需求的用户。

- **开启 MMYOLO 之旅**。本部分是第一次使用 MMYOLO 用户的必读文档，请全文仔细阅读
- **推荐专题**。本部分是 MMYOLO 中提供的以主题形式的精华文档，包括了 MMYOLO 中大量的特性等。强烈推荐使用 MMYOLO 的所有用户阅读
- **常用功能**。本部分提供了训练测试过程中用户经常会用到的各类常用功能，用户可以在用到时候再次查阅
- **实用工具**。本部分是 tools 下使用工具的汇总文档，便于大家能够快速的愉快使用 MMYOLO 中提供的各类脚本
- **基础和进阶教程**。本部分设计到 MMYOLO 中的一些基本概念和进阶教程等，适合想详细了解 MMYOLO 设计思想和结构设计的用户
- **其他**。其余部分包括 模型仓库、说明和接口文档等等

不同需求的用户可以按需选择你心怡的内容阅读。如果你对本文档有不同异议或者更好的优化办法，欢迎给 MMYOLO 提 PR ～
