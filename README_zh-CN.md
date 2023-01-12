<div align="center">
  <img src="resources/mmyolo-logo.png" width="600"/>
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
[![codecov](https://codecov.io/gh/open-mmlab/mmyolo/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmyolo)
[![license](https://img.shields.io/github/license/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/issues)

[📘使用文档](https://mmyolo.readthedocs.io/zh_CN/latest/) |
[🛠️安装教程](https://mmyolo.readthedocs.io/zh_CN/latest/get_started.html) |
[👀模型库](https://mmyolo.readthedocs.io/zh_CN/latest/model_zoo.html) |
[🆕更新日志](https://mmyolo.readthedocs.io/zh_CN/latest/notes/changelog.html) |
[🤔报告问题](https://github.com/open-mmlab/mmyolo/issues/new/choose)

</div>

<div align="center">

[English](README.md) | 简体中文

</div>

## 简介

MMYOLO 是一个基于 PyTorch 和 MMDetection 的 YOLO 系列算法开源工具箱。它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

主分支代码目前支持 PyTorch 1.6 以上的版本。
<img src="https://user-images.githubusercontent.com/45811724/190993591-bd3f1f11-1c30-4b93-b5f4-05c9ff64ff7f.gif"/>

<details open>
<summary>主要特性</summary>

- **统一便捷的算法评测**

  MMYOLO 统一了各类 YOLO 算法模块的实现, 并提供了统一的评测流程，用户可以公平便捷地进行对比分析。

- **丰富的入门和进阶文档**

  MMYOLO 提供了从入门到部署到进阶和算法解析等一系列文档，方便不同用户快速上手和扩展。

- **模块化设计**

  MMYOLO 将框架解耦成不同的模块组件，通过组合不同的模块和训练测试策略，用户可以便捷地构建自定义模型。

<img src="https://user-images.githubusercontent.com/27466624/199999337-0544a4cb-3cbd-4f3e-be26-bcd9e74db7ff.jpg" alt="基类-P5"/>
  图为 RangeKing@GitHub 提供，非常感谢！

P6 模型图详见 [model_design.md](https://github.com/open-mmlab/mmyolo/blob/dev/docs/zh_cn/algorithm_descriptions/model_design.md)。

</details>

## 最新进展

### 亮点

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

MMYOLO 中目前仅仅实现了目标检测算法，但是相比 MMDeteciton 版本有显著训练加速，训练速度相比原先版本提升 2.6 倍。

💎 **v0.3.0** 版本已经在 2023.1.8 发布：

1. 实现了 [RTMDet](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/README.md) 的快速版本。RTMDet-s 8xA100 训练只需要 14 个小时，训练速度相比原先版本提升 2.6 倍。
2. 支持 [PPYOLOE](https://github.com/open-mmlab/mmyolo/blob/main/configs/ppyoloe/README.md) 训练。
3. 支持 [YOLOv5](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/crowdhuman/yolov5_s-v61_8xb16-300e_ignore_crowdhuman.py) 的 `iscrowd` 属性训练。
4. 支持 [YOLOv5 正样本分配结果可视化](https://github.com/open-mmlab/mmyolo/blob/main/projects/assigner_visualization/README.md)
5. 新增 [YOLOv6 原理和实现全解析文档](https://github.com/open-mmlab/mmyolo/blob/main/docs/zh_cn/algorithm_descriptions/yolov6_description.md)

同时我们也推出了解读视频：

|     |                内容                |                                                                                                                                                                                                      视频                                                                                                                                                                                                       |                                                                                                         课程中的代码                                                                                                          |
| :-: | :--------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| 🌟  |            特征图可视化            |  [![Link](https://i2.hdslb.com/bfs/archive/480a0eb41fce26e0acb65f82a74501418eee1032.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV188411s7o8)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV188411s7o8)](https://www.bilibili.com/video/BV188411s7o8)  | [特征图可视化.ipynb](https://github.com/open-mmlab/OpenMMLabCourse/blob/main/codes/MMYOLO_tutorials/%5B%E5%B7%A5%E5%85%B7%E7%B1%BB%E7%AC%AC%E4%B8%80%E6%9C%9F%5D%E7%89%B9%E5%BE%81%E5%9B%BE%E5%8F%AF%E8%A7%86%E5%8C%96.ipynb) |
| 🌟  |         特征图可视化 Demo          | [![Link](http://i0.hdslb.com/bfs/archive/081f300c84d6556f40d984cfbe801fc0644ff449.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1je4y1478R/)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1je4y1478R)](https://www.bilibili.com/video/BV1je4y1478R/)  |                                                                                                                                                                                                                               |
| 🌟  |             配置全解读             |  [![Link](http://i1.hdslb.com/bfs/archive/e06daf640ea39b3c0700bb4dc758f1a253f33e13.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1214y157ck)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1214y157ck)](https://www.bilibili.com/video/BV1214y157ck)   |                                                                                   [配置全解读文档](https://zhuanlan.zhihu.com/p/577715188)                                                                                    |
| 🌟  |     源码阅读和调试「必备」技巧     |  [![Link](https://i2.hdslb.com/bfs/archive/790d2422c879ff20488910da1c4422b667ea6af7.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1N14y1V7mB)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1N14y1V7mB)](https://www.bilibili.com/video/BV1N14y1V7mB)  |                                                                           [源码阅读和调试「必备」技巧文档](https://zhuanlan.zhihu.com/p/580885852)                                                                            |
| 🌟  |          工程文件结构简析          |   [![Link](http://i2.hdslb.com/bfs/archive/41030efb84d0cada06d5451c1e6e9bccc0cdb5a3.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1LP4y117jS)[![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1LP4y117jS)](https://www.bilibili.com/video/BV1LP4y117jS)    |                                                                                [工程文件结构简析文档](https://zhuanlan.zhihu.com/p/584807195)                                                                                 |
| 🌟  |         10分钟换遍主干网络         |  [![Link](http://i0.hdslb.com/bfs/archive/c51f1aef7c605856777249a7b4478f44bd69f3bd.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1JG4y1d7GC)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1JG4y1d7GC)](https://www.bilibili.com/video/BV1JG4y1d7GC)   |     [10分钟换遍主干网络文档](https://zhuanlan.zhihu.com/p/585641598)<br>[10分钟换遍主干网络.ipynb](https://github.com/open-mmlab/OpenMMLabCourse/blob/main/codes/MMYOLO_tutorials/[实用类第二期]10分钟换遍主干网络.ipynb)     |
| 🌟  |        基于 sahi 的大图推理        | [![Link](https://i0.hdslb.com/bfs/archive/62c41f508dbcf63a4c721738171612d2d7069ac2.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1EK411R7Ws/)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1EK411R7Ws)](https://www.bilibili.com/video/BV1EK411R7Ws/) |                                   [10分钟轻松掌握大图推理.ipynb](https://github.com/open-mmlab/OpenMMLabCourse/blob/main/codes/MMYOLO_tutorials/[工具类第二期]10分钟轻松掌握大图推理.ipynb)                                   |
| 🌟  | 自定义数据集从标注到部署保姆级教程 |  [![Link](https://i2.hdslb.com/bfs/archive/13f566c89a18c9c881713b63ec14da952d4c0b14.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1RG4y137i5)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1RG4y137i5)](https://www.bilibili.com/video/BV1JG4y1d7GC)  |                                                       [自定义数据集从标注到部署保姆级教程](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/custom_dataset.html)                                                        |

发布历史和更新细节请参考 [更新日志](https://mmyolo.readthedocs.io/zh_CN/latest/notes/changelog.html)

## 安装

MMYOLO 依赖 PyTorch, MMCV, MMEngine 和 MMDetection，以下是安装的简要步骤。 更详细的安装指南请参考[安装文档](https://github.com/open-mmlab/mmyolo/blob/dev/docs/zh_cn/get_started.md)。

```shell
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install "mmengine>=0.3.1"
mim install "mmcv>=2.0.0rc1,<2.1.0"
mim install "mmdet>=3.0.0rc5,<3.1.0"
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
```

## 教程

MMYOLO 基于 MMDetection 开源库，并且采用相同的代码组织和设计方式。为了更好的使用本开源库，请先阅读 [MMDetection 概述](https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html) 对 MMDetection 进行初步地了解。

MMYOLO 用法和 MMDetection 几乎一致，所有教程都是通用的，你也可以了解 [MMDetection 用户指南和进阶指南](https://mmdetection.readthedocs.io/zh_CN/3.x/) 。

针对和 MMDetection 不同的部分，我们也准备了用户指南和进阶指南，请阅读我们的 [文档](https://mmyolo.readthedocs.io/zh_CN/latest/) 。

- 用户指南

  - [训练 & 测试](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/index.html)
    - [学习 YOLOv5 配置文件](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/config.html)
  - [从入门到部署全流程](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/index.html#从入门到部署全流程)
    - [自定义数据集](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/custom_dataset.html)
    - [YOLOv5 从入门到部署全流程](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/yolov5_tutorial.html)
  - [实用工具](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/index.html#实用工具)
    - [可视化教程](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/visualization.html)
    - [实用工具](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html)

- 算法描述

  - [必备基础](https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/index.html#基础内容)
    - [模型设计相关说明](https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/model_design.html)
  - [算法原理和实现全解析](https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/index.html#算法原理和实现全解析)
    - [YOLOv5 原理和实现全解析](https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/yolov5_description.html)
    - [YOLOv6 原理和实现全解析](https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/yolov6_description.html)
    - [RTMDet 原理和实现全解析](https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/rtmdet_description.html)

- 算法部署

  - [部署必备教程](https://mmyolo.readthedocs.io/zh_CN/latest/deploy/index.html#id1)
    - [部署必备教程](https://mmyolo.readthedocs.io/zh_CN/latest/deploy/basic_deployment_guide.html)
  - [部署全流程说明](https://mmyolo.readthedocs.io/zh_CN/latest/deploy/index.html#id2)
    - [YOLOv5 部署全流程说明](https://mmyolo.readthedocs.io/zh_CN/latest/deploy/yolov5_deployment.html)

- 进阶指南

  - [模块组合](https://mmyolo.readthedocs.io/zh_CN/latest/advanced_guides/module_combination.html)
  - [数据流](https://mmyolo.readthedocs.io/zh_CN/latest/advanced_guides/data_flow.html)
  - [How to](https://mmyolo.readthedocs.io/zh_CN/latest/advanced_guides/how_to.html)
  - [插件](https://mmyolo.readthedocs.io/zh_CN/latest/advanced_guides/plugins.html)

- [解读文章和资源汇总](https://mmyolo.readthedocs.io/zh_CN/latest/article.html)

## 基准测试和模型库

测试结果和模型可以在 [模型库](https://mmyolo.readthedocs.io/zh_CN/latest//model_zoo.html) 中找到。

<details open>
<summary><b>支持的算法</b></summary>

- [x] [YOLOv5](configs/yolov5)
- [x] [YOLOX](configs/yolox)
- [x] [RTMDet](configs/rtmdet)
- [x] [YOLOv6](configs/yolov6)
- [x] [YOLOv7](configs/yolov7)
- [x] [PPYOLOE](configs/ppyoloe)
- [ ] [YOLOv8](configs/yolov8)（仅推理）

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
        <li>YOLOXCSPDarknet</li>
        <li>EfficientRep</li>
        <li>CSPNeXt</li>
        <li>YOLOv7Backbone</li>
        <li>PPYOLOECSPResNet</li>
      </ul>
      </td>
      <td>
      <ul>
        <li>YOLOv5PAFPN</li>
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

## 常见问题

请参考 [FAQ](https://mmyolo.readthedocs.io/zh_CN/latest/notes/faq.html) 了解其他用户的常见问题。

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMYOLO 所作出的努力。我们将正在进行中的项目添加进了[GitHub Projects](https://github.com/open-mmlab/mmyolo/projects)页面，非常欢迎社区用户能参与进这些项目中来。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

MMYOLO 是一款由来自不同高校和企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望这个工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现已有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 引用

如果你觉得本项目对你的研究工作有所帮助，请参考如下 bibtex 引用 MMYOLO

```latex
@misc{mmyolo2022,
    title={{MMYOLO: OpenMMLab YOLO} series toolbox and benchmark},
    author={MMYOLO Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmyolo}},
    year={2022}
}
```

## 开源许可证

该项目采用 [GPL 3.0 开源许可证](LICENSE)。

## OpenMMLab 的其他项目

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab 深度学习模型训练基础库
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
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
- [MMEval](https://github.com/open-mmlab/mmeval): OpenMMLab 机器学习算法评测库

## 欢迎加入 OpenMMLab 社区

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
