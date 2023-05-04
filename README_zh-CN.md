<div align="center">
  <img src="https://user-images.githubusercontent.com/27466624/222385182-1247251c-8fac-4e77-94f5-57580e0ce3bd.png" width="100%"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmyolo)](https://pypi.org/project/mmyolo)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmyolo.readthedocs.io/zh_CN/latest/)
[![deploy](https://github.com/open-mmlab/mmyolo/workflows/deploy/badge.svg)](https://github.com/open-mmlab/mmyolo/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmyolo/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmyolo)
[![license](https://img.shields.io/github/license/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/issues)

[📘使用文档](https://mmyolo.readthedocs.io/zh_CN/latest/) |
[🛠️安装教程](https://mmyolo.readthedocs.io/zh_CN/latest/get_started/installation.html) |
[👀模型库](https://mmyolo.readthedocs.io/zh_CN/latest/model_zoo.html) |
[🆕更新日志](https://mmyolo.readthedocs.io/zh_CN/latest/notes/changelog.html) |
[🤔报告问题](https://github.com/open-mmlab/mmyolo/issues/new/choose)

</div>

<div align="center">

[English](README.md) | 简体中文

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1046608014234370059" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## 📄 Table of Contents

- [🥳 🚀 最新进展](#--最新进展-)
  - [✨ 亮点](#-亮点-)
- [📖 简介](#-简介-)
- [🛠️ 安装](#️%EF%B8%8F-安装-)
- [👨‍🏫 教程](#-教程-)
- [📊 基准测试和模型库](#-基准测试和模型库-)
- [❓ 常见问题](#-常见问题-)
- [🙌 贡献指南](#-贡献指南-)
- [🤝 致谢](#🤝-致谢-)
- [🖊️ 引用](#️-引用-)
- [🎫 开源许可证](#-开源许可证-)
- [🏗️ OpenMMLab 的其他项目](#%EF%B8%8F-openmmlab-的其他项目-)
- [❤️ 欢迎加入 OpenMMLab 社区](#%EF%B8%8F-欢迎加入-openmmlab-社区-)

## 🥳 🚀 最新进展 [🔝](#-table-of-contents)

💎 **v0.5.0** 版本已经在 2023.3.2 发布：

1. 支持了 [RTMDet-R](https://github.com/open-mmlab/mmyolo/blob/dev/configs/rtmdet/README.md#rotated-object-detection) 旋转框目标检测任务和算法
2. [YOLOv8](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov8/README.md) 支持使用 mask 标注提升目标检测模型性能
3. 支持 [MMRazor](https://github.com/open-mmlab/mmyolo/blob/dev/configs/razor/subnets/README.md) 搜索的 NAS 子网络作为 YOLO 系列算法的 backbone
4. 支持调用 [MMRazor](https://github.com/open-mmlab/mmyolo/blob/dev/configs/rtmdet/distillation/README.md) 对 RTMDet 进行知识蒸馏
5. [MMYOLO](https://mmyolo.readthedocs.io/zh_CN/dev/) 文档结构优化，内容全面升级
6. 基于 RTMDet 训练超参提升 YOLOX 精度和训练速度
7. 支持模型参数量、FLOPs 计算和提供 T4 设备上 GPU 延时数据，并更新了 [Model Zoo](https://github.com/open-mmlab/mmyolo/blob/dev/docs/zh_cn/model_zoo.md)
8. 支持测试时增强 TTA
9. 支持 RTMDet、YOLOv8 和 YOLOv7 assigner 可视化

我们提供了实用的**脚本命令速查表**

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/213104312-3580c783-2423-442f-b5f6-79204a06adb5.png">
</div>

你可以点击[链接](https://pan.baidu.com/s/1QEaqT7YayUdEvh1an0gjHg?pwd=yolo)，下载高清版 PDF 文件。

同时我们也推出了解读视频：

|     |                内容                |                                                                                                                                                                                                     视频                                                                                                                                                                                                      |                                                                                                         课程中的代码                                                                                                          |
| :-: | :--------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| 🌟  |            特征图可视化            | [![Link](https://i2.hdslb.com/bfs/archive/480a0eb41fce26e0acb65f82a74501418eee1032.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV188411s7o8)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV188411s7o8)](https://www.bilibili.com/video/BV188411s7o8) | [特征图可视化.ipynb](https://github.com/open-mmlab/OpenMMLabCourse/blob/main/codes/MMYOLO_tutorials/%5B%E5%B7%A5%E5%85%B7%E7%B1%BB%E7%AC%AC%E4%B8%80%E6%9C%9F%5D%E7%89%B9%E5%BE%81%E5%9B%BE%E5%8F%AF%E8%A7%86%E5%8C%96.ipynb) |
| 🌟  |     源码阅读和调试「必备」技巧     | [![Link](https://i2.hdslb.com/bfs/archive/790d2422c879ff20488910da1c4422b667ea6af7.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1N14y1V7mB)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1N14y1V7mB)](https://www.bilibili.com/video/BV1N14y1V7mB) |                                                                           [源码阅读和调试「必备」技巧文档](https://zhuanlan.zhihu.com/p/580885852)                                                                            |
| 🌟  |         10分钟换遍主干网络         | [![Link](http://i0.hdslb.com/bfs/archive/c51f1aef7c605856777249a7b4478f44bd69f3bd.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1JG4y1d7GC)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1JG4y1d7GC)](https://www.bilibili.com/video/BV1JG4y1d7GC)  |     [10分钟换遍主干网络文档](https://zhuanlan.zhihu.com/p/585641598)<br>[10分钟换遍主干网络.ipynb](https://github.com/open-mmlab/OpenMMLabCourse/blob/main/codes/MMYOLO_tutorials/[实用类第二期]10分钟换遍主干网络.ipynb)     |
| 🌟  | 自定义数据集从标注到部署保姆级教程 | [![Link](https://i2.hdslb.com/bfs/archive/13f566c89a18c9c881713b63ec14da952d4c0b14.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1RG4y137i5)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1RG4y137i5)](https://www.bilibili.com/video/BV1JG4y1d7GC) |                                                 [自定义数据集从标注到部署保姆级教程](https://github.com/open-mmlab/mmyolo/blob/dev/docs/zh_cn/user_guides/custom_dataset.md)                                                  |
| 🌟  |      顶会第一步 · 模块自定义       | [![Link](http://i2.hdslb.com/bfs/archive/5b23d41ac57466824eaf185ef806ef734414e93b.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1yd4y1j7VD)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1yd4y1j7VD)](https://www.bilibili.com/video/BV1yd4y1j7VD)  |                                    [顶会第一步·模块自定义.ipynb](https://github.com/open-mmlab/OpenMMLabCourse/blob/main/codes/MMYOLO_tutorials/[实用类第四期]顶会第一步·模块自定义.ipynb)                                    |

完整视频列表请参考 [中文解读资源汇总 - 视频](https://mmyolo.readthedocs.io/zh_CN/latest/get_started/article.html)

发布历史和更新细节请参考 [更新日志](https://mmyolo.readthedocs.io/zh_CN/latest/notes/changelog.html)

### ✨ 亮点 [🔝](#-table-of-contents)

我们很高兴向大家介绍我们在实时目标识别任务方面的最新成果 RTMDet，包含了一系列的全卷积单阶段检测模型。 RTMDet 不仅在从 tiny 到 extra-large 尺寸的目标检测模型上实现了最佳的参数量和精度的平衡，而且在实时实例分割和旋转目标检测任务上取得了最先进的成果。 更多细节请参阅[技术报告](https://arxiv.org/abs/2212.07784)。 预训练模型可以在[这里](configs/rtmdet)找到。

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/real-time-instance-segmentation-on-mscoco)](https://paperswithcode.com/sota/real-time-instance-segmentation-on-mscoco?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-hrsc2016)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-hrsc2016?p=rtmdet-an-empirical-study-of-designing-real)

| Task                     | Dataset | AP                                   | FPS(TRT FP16 BS1 3090) |
| ------------------------ | ------- | ------------------------------------ | ---------------------- |
| Object Detection         | COCO    | 52.8                                 | 322                    |
| Instance Segmentation    | COCO    | 44.6                                 | 188                    |
| Rotated Object Detection | DOTA    | 78.9(single-scale)/81.3(multi-scale) | 121                    |

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/208044554-1e8de6b5-48d8-44e4-a7b5-75076c7ebb71.png"/>
</div>

MMYOLO 中目前实现了目标检测和旋转框目标检测算法，但是相比 MMDeteciton 版本有显著训练加速，训练速度相比原先版本提升 2.6 倍。

## 📖 简介 [🔝](#-table-of-contents)

MMYOLO 是一个基于 PyTorch 和 MMDetection 的 YOLO 系列算法开源工具箱。它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

主分支代码目前支持 PyTorch 1.6 以上的版本。
<img src="https://user-images.githubusercontent.com/45811724/190993591-bd3f1f11-1c30-4b93-b5f4-05c9ff64ff7f.gif"/>

<details open>
<summary>主要特性</summary>

- 🕹️ **统一便捷的算法评测**

  MMYOLO 统一了各类 YOLO 算法模块的实现, 并提供了统一的评测流程，用户可以公平便捷地进行对比分析。

- 📚 **丰富的入门和进阶文档**

  MMYOLO 提供了从入门到部署到进阶和算法解析等一系列文档，方便不同用户快速上手和扩展。

- 🧩 **模块化设计**

  MMYOLO 将框架解耦成不同的模块组件，通过组合不同的模块和训练测试策略，用户可以便捷地构建自定义模型。

<img src="https://user-images.githubusercontent.com/27466624/199999337-0544a4cb-3cbd-4f3e-be26-bcd9e74db7ff.jpg" alt="基类-P5"/>
  图为 RangeKing@GitHub 提供，非常感谢！

P6 模型图详见 [model_design.md](docs/zh_cn/recommended_topics/model_design.md)。

</details>

## 🛠️ 安装 [🔝](#-table-of-contents)

MMYOLO 依赖 PyTorch, MMCV, MMEngine 和 MMDetection，以下是安装的简要步骤。 更详细的安装指南请参考[安装文档](docs/zh_cn/get_started/installation.md)。

```shell
conda create -n mmyolo python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate mmyolo
pip install openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0rc6,<3.1.0"
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
```

## 👨‍🏫 教程 [🔝](#-table-of-contents)

MMYOLO 基于 MMDetection 开源库，并且采用相同的代码组织和设计方式。为了更好的使用本开源库，请先阅读 [MMDetection 概述](https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html) 对 MMDetection 进行初步地了解。

MMYOLO 用法和 MMDetection 几乎一致，所有教程都是通用的，你也可以了解 [MMDetection 用户指南和进阶指南](https://mmdetection.readthedocs.io/zh_CN/3.x/) 。

针对和 MMDetection 不同的部分，我们也准备了用户指南和进阶指南，请阅读我们的 [文档](https://mmyolo.readthedocs.io/zh_CN/latest/) 。

<details>
<summary>开启 MMYOLO 之旅</summary>

- [概述](docs/zh_cn/get_started/overview.md)
- [依赖](docs/zh_cn/get_started/dependencies.md)
- [安装和验证](docs/zh_cn/get_started/installation.md)
- [15 分钟上手 MMYOLO 目标检测](docs/zh_cn/get_started/15_minutes_object_detection.md)
- [15 分钟上手 MMYOLO 旋转框目标检测](docs/zh_cn/get_started/15_minutes_rotated_object_detection.md)
- [15 分钟上手 MMYOLO 实例分割](docs/zh_cn/get_started/15_minutes_instance_segmentation.md)
- [中文解读资源汇总](docs/zh_cn/get_started/article.md)

</details>

<details>
<summary>推荐专题</summary>

- [如何给 MMYOLO 贡献代码](docs/zh_cn/recommended_topics/contributing.md)
- [训练和测试技巧](docs/zh_cn/recommended_topics/training_testing_tricks.md)
- [MMYOLO 模型结构设计](docs/zh_cn/recommended_topics/model_design.md)
- [原理和实现全解析](docs/zh_cn/recommended_topics/algorithm_descriptions/)
- [轻松更换主干网络](docs/zh_cn/recommended_topics/replace_backbone.md)
- [MMYOLO 模型复杂度分析](docs/zh_cn/recommended_topics/complexity_analysis.md)
- [标注+训练+测试+部署全流程](docs/zh_cn/recommended_topics/labeling_to_deployment_tutorials.md)
- [关于可视化的一切](docs/zh_cn/recommended_topics/visualization.md)
- [模型部署流程](docs/zh_cn/recommended_topics/deploy/)
- [常见错误排查步骤](docs/zh_cn/recommended_topics/troubleshooting_steps.md)
- [MMYOLO 应用范例介绍](docs/zh_cn/recommended_topics/application_examples/)
- [MM 系列 Repo 必备基础](docs/zh_cn/recommended_topics/mm_basics.md)
- [数据集准备和说明](docs/zh_cn/recommended_topics/dataset_preparation.md)

</details>

<details>
<summary>常用功能</summary>

- [恢复训练](docs/zh_cn/common_usage/resume_training.md)
- [开启和关闭 SyncBatchNorm](docs/zh_cn/common_usage/syncbn.md)
- [开启混合精度训练](docs/zh_cn/common_usage/amp_training.md)
- [多尺度训练和测试](docs/zh_cn/common_usage/ms_training_testing.md)
- [测试时增强相关说明](docs/zh_cn/common_usage/tta.md)
- [给主干网络增加插件](docs/zh_cn/common_usage/plugins.md)
- [冻结指定网络层权重](docs/zh_cn/common_usage/freeze_layers.md)
- [输出模型预测结果](docs/zh_cn/common_usage/output_predictions.md)
- [设置随机种子](docs/zh_cn/common_usage/set_random_seed.md)
- [算法组合替换教程](docs/zh_cn/common_usage/module_combination.md)
- [使用 mim 跨库调用其他 OpenMMLab 仓库的脚本](docs/zh_cn/common_usage/mim_usage.md)
- [应用多个 Neck](docs/zh_cn/common_usage/multi_necks.md)
- [指定特定设备训练或推理](docs/zh_cn/common_usage/specify_device.md)
- [单通道和多通道应用案例](docs/zh_cn/common_usage/single_multi_channel_applications.md)
- [MM 系列开源库注册表](docs/zh_cn/common_usage/registries_info.md)

</details>

<details>
<summary>实用工具</summary>

- [可视化 COCO 标签](docs/zh_cn/useful_tools/browse_coco_json.md)
- [可视化数据集](docs/zh_cn/useful_tools/browse_dataset.md)
- [打印完整配置文件](docs/zh_cn/useful_tools/print_config.md)
- [可视化数据集分析结果](docs/zh_cn/useful_tools/dataset_analysis.md)
- [优化锚框尺寸](docs/zh_cn/useful_tools/optimize_anchors.md)
- [提取 COCO 子集](docs/zh_cn/useful_tools/extract_subcoco.md)
- [可视化优化器参数策略](docs/zh_cn/useful_tools/vis_scheduler.md)
- [数据集转换](docs/zh_cn/useful_tools/dataset_converters.md)
- [数据集下载](docs/zh_cn/useful_tools/download_dataset.md)
- [日志分析](docs/zh_cn/useful_tools/log_analysis.md)
- [模型转换](docs/zh_cn/useful_tools/model_converters.md)

</details>

<details>
<summary>基础教程</summary>

- [学习 YOLOv5 配置文件](docs/zh_cn/tutorials/config.md)
- [数据流](docs/zh_cn/tutorials/data_flow.md)
- [旋转目标检测](docs/zh_cn/tutorials/rotated_detection.md)
- [自定义安装](docs/zh_cn/tutorials/custom_installation.md)
- [常见警告说明](docs/zh_cn/tutorials/warning_notes.md)
- [常见问题](docs/zh_cn/tutorials/faq.md)

</details>

<details>
<summary>进阶教程</summary>

- [MMYOLO 跨库应用解析](docs/zh_cn/advanced_guides/cross-library_application.md)

</details>

<details>
<summary>说明</summary>

- [更新日志](docs/zh_cn/notes/changelog.md)
- [兼容性说明](docs/zh_cn/notes/compatibility.md)
- [默认约定](docs/zh_cn/notes/conventions.md)
- [代码规范](docs/zh_cn/notes/code_style.md)

</details>

## 📊 基准测试和模型库 [🔝](#-table-of-contents)

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/222087414-168175cc-dae6-4c5c-a8e3-3109a152dd19.png"/>
</div>

测试结果和模型可以在 [模型库](docs/zh_cn/model_zoo.md) 中找到。

<details open>
<summary><b>支持的任务</b></summary>

- [x] 目标检测
- [x] 旋转框目标检测

</details>

<details open>
<summary><b>支持的算法</b></summary>

- [x] [YOLOv5](configs/yolov5)
- [ ] [YOLOv5u](configs/yolov5/yolov5u) (仅推理)
- [x] [YOLOX](configs/yolox)
- [x] [RTMDet](configs/rtmdet)
- [x] [RTMDet-Rotated](configs/rtmdet)
- [x] [YOLOv6](configs/yolov6)
- [x] [YOLOv7](configs/yolov7)
- [x] [PPYOLOE](configs/ppyoloe)
- [x] [YOLOv8](configs/yolov8)

</details>

<details open>
<summary><b>支持的数据集</b></summary>

- [x] COCO Dataset
- [x] VOC Dataset
- [x] CrowdHuman Dataset
- [x] DOTA 1.0 Dataset

</details>

<details open>
<div align="center">
  <b>模块组件</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Loss</b>
      </td>
      <td>
        <b>Common</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>YOLOv5CSPDarknet</li>
        <li>YOLOv8CSPDarknet</li>
        <li>YOLOXCSPDarknet</li>
        <li>EfficientRep</li>
        <li>CSPNeXt</li>
        <li>YOLOv7Backbone</li>
        <li>PPYOLOECSPResNet</li>
        <li>mmdet backbone</li>
        <li>mmcls backbone</li>
        <li>timm</li>
      </ul>
      </td>
      <td>
      <ul>
        <li>YOLOv5PAFPN</li>
        <li>YOLOv8PAFPN</li>
        <li>YOLOv6RepPAFPN</li>
        <li>YOLOXPAFPN</li>
        <li>CSPNeXtPAFPN</li>
        <li>YOLOv7PAFPN</li>
        <li>PPYOLOECSPPAFPN</li>
      </ul>
      </td>
      <td>
        <ul>
          <li>IoULoss</li>
          <li>mmdet loss</li>
        </ul>
      </td>
      <td>
        <ul>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

</details>

## ❓ 常见问题 [🔝](#-table-of-contents)

请参考 [FAQ](docs/zh_cn/tutorials/faq.md) 了解其他用户的常见问题。

## 🙌 贡献指南 [🔝](#-table-of-contents)

我们感谢所有的贡献者为改进和提升 MMYOLO 所作出的努力。我们将正在进行中的项目添加进了[GitHub Projects](https://github.com/open-mmlab/mmyolo/projects)页面，非常欢迎社区用户能参与进这些项目中来。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 🤝 致谢 [🔝](#-table-of-contents)

MMYOLO 是一款由来自不同高校和企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望这个工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现已有算法并开发自己的新模型，从而不断为开源社区提供贡献。

<div align="center">
  <a href="https://github.com/open-mmlab/mmyolo/graphs/contributors"><img src="https://contrib.rocks/image?repo=open-mmlab/mmyolo"/></a>
</div>

## 🖊️ 引用 [🔝](#-table-of-contents)

如果你觉得本项目对你的研究工作有所帮助，请参考如下 bibtex 引用 MMYOLO

```latex
@misc{mmyolo2022,
    title={{MMYOLO: OpenMMLab YOLO} series toolbox and benchmark},
    author={MMYOLO Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmyolo}},
    year={2022}
}
```

## 🎫 开源许可证 [🔝](#-table-of-contents)

该项目采用 [GPL 3.0 开源许可证](LICENSE)。

## 🏗️ OpenMMLab 的其他项目 [🔝](#-table-of-contents)

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab 深度学习模型训练基础库
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab 深度学习预训练工具箱
- [MMagic](https://github.com/open-mmlab/mmagic): OpenMMLab 新一代人工智能内容生成（AIGC）工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO 系列工具箱
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMEval](https://github.com/open-mmlab/mmeval): OpenMMLab 机器学习算法评测库
- [Playground](https://github.com/open-mmlab/playground): 收集和展示 OpenMMLab 相关的前沿、有趣的社区项目

## ❤️ 欢迎加入 OpenMMLab 社区 [🔝](#-table-of-contents)

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)

<div align="center">
<img src="resources/zhihu_qrcode.jpg" height="400" />  <img src="resources/qq_group_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
