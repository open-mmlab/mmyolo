# 概述

本章向您介绍 MMYOLO 的整体框架，并提供详细的教程链接。

## 什么是 MMYOLO

<div align=center>
<img src="https://user-images.githubusercontent.com/45811724/190993591-bd3f1f11-1c30-4b93-b5f4-05c9ff64ff7f.gif" alt="图片"/>
</div>

MMYOLO 是一个 YOLO 系列的算法工具箱，目前仅实现了目标检测任务，**后续**会支持实例分割、全景分割和关键点检测等多种任务。其包括丰富的目标检测算法以及相关的组件和模块，下面是它的整体框架：

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

## 如何使用本指南

以下是 MMYOLO 的详细指南：

1. 安装说明见[开始你的第一步](get_started.md)

2. MMYOLO 的基本使用方法请参考以下教程：

   - [训练和测试](https://mmyolo.readthedocs.io/zh_cn/latest/user_guides/index.html#训练-测试)
   - [从入门到部署全流程](https://mmyolo.readthedocs.io/zh_cn/latest/user_guides/index.html#从入门到部署全流程)
   - [实用工具](https://mmyolo.readthedocs.io/zh_cn/latest/user_guides/index.html#实用工具)

3. YOLO 系列算法实现和全解析教程：

   - [必备基础](https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/index.html#必备基础)
   - [原理和实现全解析](https://mmyolo.readthedocs.io/zh_cn/latest/algorithm_descriptions/index.html#原理和实现全解析)

4. 参考以下教程深入了解：

   - [数据流](https://mmyolo.readthedocs.io/zh_cn/latest/advanced_guides/index.html#数据流)
   - [How to](https://mmyolo.readthedocs.io/zh_cn/latest/advanced_guides/index.html#how-to)

5. [解读文章和资源汇总](article.md)
