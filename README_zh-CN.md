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
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmyolo.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmyolo/workflows/build/badge.svg)](https://github.com/open-mmlab/mmyolo/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmyolo/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmyolo)
[![license](https://img.shields.io/github/license/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/issues)

[📘使用文档](https://mmyolo.readthedocs.io/zh_CN/latest/) |
[🛠️安装教程](https://mmyolo.readthedocs.io/zh_CN/latest/get_started.html) |
[👀模型库](https://mmyolo.readthedocs.io/zh_CN/latest/model_zoo.html) |
[🆕更新日志](https://mmyolo.readthedocs.io/en/latest/notes/changelog.html) |
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

<img src="https://user-images.githubusercontent.com/27466624/190986949-01414a91-baae-4228-8828-c59db58dcf36.jpg" alt="基类"/>
  图为 RangeKing@GitHub 提供，非常感谢！

</details>

## 最新进展

**v0.1.1** 版本已经在 2022.9.29 发布：

- 支持了 [RTMDet](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet)
- 新增了 [RTMDet 原理和实现全解析中文文档](https://github.com/open-mmlab/mmyolo/blob/main/docs/zh_cn/algorithm_descriptions/rtmdet_description.md)
- 支持对 backbone 自定义插件，并且更新了 How-to 文档

发布历史和更新细节请参考 [更新日志](https://mmyolo.readthedocs.io/zh_CN/latest/notes/changelog.html)

## 安装

MMYOLO 依赖 PyTorch, MMCV, MMEngine 和 MMDetection，以下是安装的简要步骤。 更详细的安装指南请参考[安装文档](docs/zh_cn/get_started.md)。

```shell
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0rc1,<2.1.0"
mim install "mmdet>=3.0.0rc1,<3.1.0"
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

  - [训练 & 测试](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/index.html#训练-测试)
    - [学习 YOLOv5 配置文件](docs/zh_cn/user_guides/config.md)
  - [从入门到部署全流程](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/index.html#从入门到部署全流程)
    - [YOLOv5 从入门到部署全流程](docs/zh_cn/user_guides/yolov5_tutorial.md)
  - [实用工具](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/index.html#实用工具)
    - [可视化教程](docs/zh_cn/user_guides/visualization.md)
    - [实用工具](docs/zh_cn/user_guides/useful_tools.md)

- 算法描述

  - [必备基础](https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/index.html#基础内容)
    - [模型设计相关说明](docs/zh_cn/algorithm_descriptions/model_design.md)
  - [算法原理和实现全解析](https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/index.html#算法原理和实现全解析)
    - [YOLOv5 原理和实现全解析](docs/zh_cn/algorithm_descriptions/yolov5_description.md)
    - [RTMDet 原理和实现全解析](docs/zh_cn/algorithm_descriptions/rtmdet_description.md)

- 进阶指南

  - [How to](https://mmyolo.readthedocs.io/zh_CN/3.x/advanced_guides/index.html#how-to)

## 基准测试和模型库

测试结果和模型可以在 [模型库](docs/zh_cn/model_zoo.md) 中找到。

<details open>
<summary><b>支持的算法</b></summary>

- [x] [YOLOv5](configs/yolov5)
- [x] [YOLOX](configs/yolox)
- [x] [RTMDet](configs/rtmdet)
- [ ] [YOLOv6](configs/yolov6)(仅推理)

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
      </ul>
      </td>
      <td>
      <ul>
        <li>YOLOv5PAFPN</li>
        <li>YOLOv6RepPAFPN</li>
        <li>YOLOXPAFPN</li>
        <li>CSPNeXtPAFPN</li>
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

请参考 [FAQ](docs/zh_cn/notes/faq.md) 了解其他用户的常见问题。

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
