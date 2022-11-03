<div align="center">
  <img src="resources/mmyolo-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmyolo)](https://pypi.org/project/mmyolo)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmyolo.readthedocs.io/en/latest/)
[![deploy](https://github.com/open-mmlab/mmyolo/workflows/deploy/badge.svg)](https://github.com/open-mmlab/mmyolo/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmyolo/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmyolo)
[![license](https://img.shields.io/github/license/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmyolo.svg)](https://github.com/open-mmlab/mmyolo/issues)

[📘Documentation](https://mmyolo.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmyolo.readthedocs.io/en/latest/get_started.html) |
[👀Model Zoo](https://mmyolo.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmyolo.readthedocs.io/en/latest/notes/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmyolo/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

MMYOLO is an open source toolbox for YOLO series algorithms based on PyTorch and [MMDetection](https://github.com/open-mmlab/mmdetection). It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.6+**.
<img src="https://user-images.githubusercontent.com/45811724/190993591-bd3f1f11-1c30-4b93-b5f4-05c9ff64ff7f.gif"/>

<details open>
<summary>Major features</summary>

- **Unified and convenient benchmark**

  MMYOLO unifies the implementation of modules in various YOLO algorithms and provides a unified benchmark. Users can compare and analyze in a fair and convenient way.

- **Rich and detailed documentation**

  MMYOLO provides rich documentation for getting started, model deployment, advanced usages, and algorithm analysis, making it easy for users at different levels to get started and make extensions quickly.

- **Modular Design**

  MMYOLO decomposes the framework into different components where users can easily customize a model by combining different modules with various training and testing strategies.

<img src="https://user-images.githubusercontent.com/27466624/190986949-01414a91-baae-4228-8828-c59db58dcf36.jpg" alt="BaseModule"/>
  The figure is contributed by RangeKing@GitHub, thank you very much!

</details>

## What's New

💎 **v0.1.2** was released on 3/11/2022:

1. Support [YOLOv5/YOLOv6/YOLOX/RTMDet deployments](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy) for ONNXRuntime and TensorRT
2. Support [YOLOv6](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov6) s/t/n model training
3. YOLOv5 supports [P6 model training which can input 1280-scale images](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5)
4. YOLOv5 supports [VOC dataset training](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/voc)
5. Support [PPYOLOE](https://github.com/open-mmlab/mmyolo/blob/main/configs/ppyoloe) and [YOLOv7](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov7) model inference and official weight conversion
6. Add YOLOv5 replacement [backbone tutorial](https://github.com/open-mmlab/mmyolo/blob/dev/docs/en/advanced_guides/how_to.md#use-backbone-network-implemented-in-other-openmmlab-repositories) in How-to documentation

For release history and update details, please refer to [changelog](https://mmyolo.readthedocs.io/en/latest/notes/changelog.html).

## Installation

MMYOLO relies on PyTorch, MMCV, MMEngine, and MMDetection. Below are quick steps for installation. Please refer to the [Install Guide](docs/en/get_started.md) for more detailed instructions.

```shell
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install "mmengine>=0.3.0"
mim install "mmcv>=2.0.0rc1,<2.1.0"
mim install "mmdet>=3.0.0rc2,<3.1.0"
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
```

## Tutorial

MMYOLO is based on MMDetection and adopts the same code structure and design approach. To get better use of this, please read [MMDetection Overview](https://mmdetection.readthedocs.io/en/latest/get_started.html) for the first understanding of MMDetection.

The usage of MMYOLO is almost identical to MMDetection and all tutorials are straightforward to use, you can also learn about [MMDetection User Guide and Advanced Guide](https://mmdetection.readthedocs.io/en/3.x/).

For different parts from MMDetection, we have also prepared user guides and advanced guides, please read our [documentation](https://mmyolo.readthedocs.io/zenh_CN/latest/).

- User Guides

  - [Train & Test](https://mmyolo.readthedocs.io/en/latest/user_guides/index.html#train-test)
    - [Learn about Configs with YOLOv5](docs/en/user_guides/config.md)
  - [From getting started to deployment](https://mmyolo.readthedocs.io/en/latest/user_guides/index.html#get-started-to-deployment)
    - [From getting started to deployment with YOLOv5](docs/en/user_guides/yolov5_tutorial.md)
  - [Useful Tools](https://mmdetection.readthedocs.io/en/latest/user_guides/index.html#useful-tools)
    - [Visualization](docs/en/user_guides/visualization.md)
    - [Useful Tools](docs/en/user_guides/useful_tools.md)

- Algorithm description

  - [Essential Basics](https://mmyolo.readthedocs.io/en/latest/algorithm_descriptions/index.html#essential-basics)
    - [Model design-related instructions](docs/en/algorithm_descriptions/model_design.md)
  - [Algorithm principles and implementation](https://mmyolo.readthedocs.io/en/latest/algorithm_descriptions/index.html#algorithm-principles-and-implementation)
    - [Algorithm principles and implementation with YOLOv5](docs/en/algorithm_descriptions/yolov5_description.md)

- Deployment Guides

  - [Basic Deployment Guide](https://mmyolo.readthedocs.io/en/latest/algorithm_descriptions/index.html#basic-deployment-guide)
    - [Basic Deployment Guide](docs/en/deploy/basic_deployment_guide.md)
  - [Deployment Tutorial](https://mmyolo.readthedocs.io/en/latest/algorithm_descriptions/index.html#deployment-tutorial)
    - [YOLOv5 Deployment](docs/en/deploy/yolov5_deployment.md)

- Advanced Guides

  - [Data flow](docs/en/advanced_guides/data_flow.md)
  - [How to](docs/en/advanced_guides/how_to.md)

## Overview of Benchmark and Model Zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

<details open>
<summary><b>Supported Algorithms</b></summary>

- [x] [YOLOv5](configs/yolov5)
- [x] [YOLOX](configs/yolox)
- [x] [RTMDet](configs/rtmdet)
- [x] [YOLOv6](configs/yolov6)
- [ ] [PPYOLOE](configs/ppyoloe)(Inference only)
- [ ] [YOLOv7](configs/yolov7)(Inference only)

</details>

<details open>
<div align="center">
  <b>Module Components</b>
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

## FAQ

Please refer to the [FAQ](docs/en/notes/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improving MMYOLO. Ongoing projects can be found in our [GitHub Projects](https://github.com/open-mmlab/mmyolo/projects). Welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMYOLO is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedback.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you find this project useful in your research, please consider cite:

```latex
@misc{mmyolo2022,
    title={{MMYOLO: OpenMMLab YOLO} series toolbox and benchmark},
    author={MMYOLO Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmyolo}},
    year={2022}
}
```

## License

This project is released under the [GPL 3.0 license](LICENSE).

## Projects in OpenMMLab

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
- [MMEval](https://github.com/open-mmlab/mmeval): OpenMMLab machine learning evaluation library.
