# Overview

## MMYOLO Introduction

<div align=center>
<img src="https://user-images.githubusercontent.com/45811724/190993591-bd3f1f11-1c30-4b93-b5f4-05c9ff64ff7f.gif" alt="image"/>
</div>

MMYOLO is an open-source algorithms toolkit of YOLO based on PyTorch and MMDetection, part of the [OpenMMLab](https://openmmlab.com/) project. MMYOLO is positioned as a popular open-source library of YOLO series and core library of industrial applications. Its vision diagram is shown as follows:

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/220060451-d50414e5-a239-45b7-a4db-ed8699820300.png" alt="vision diagram"/>
</div>

The following tasks are currently supported:

<details open>
<summary><b>Tasks currently supported</b></summary>

- Object detection
- Rotated object detection

</details>

The YOLO series of algorithms currently supported are as follows:

<details open>
<summary><b>Algorithms currently supported</b></summary>

- YOLOv5
- YOLOX
- RTMDet
- RTMDet-Rotated
- YOLOv6
- YOLOv7
- PPYOLOE
- YOLOv8

</details>

The datasets currently supported are as follows:

<details open>
<summary><b>Datasets currently supported</b></summary>

- COCO Dataset
- VOC Dataset
- CrowdHuman Dataset
- DOTA 1.0 Dataset

</details>

MMYOLO runs on Linux, Windows, macOS, and supports PyTorch 1.7 or later. It has the following three characteristics:

- 🕹️ **Unified and convenient algorithm evaluation**

  MMYOLO unifies various YOLO algorithm modules and provides a unified evaluation process, so that users can compare and analyze fairly and conveniently.

- 📚 **Extensive documentation for started and advanced**

  MMYOLO provides a series of documents, including getting started, deployment, advanced practice and algorithm analysis, which is convenient for different users to get started and expand.

- 🧩 **Modular Design**

  MMYOLO disentangled the framework into modular components, and users can easily build custom models by combining different modules and training and testing strategies.

<img src="https://user-images.githubusercontent.com/27466624/199999337-0544a4cb-3cbd-4f3e-be26-bcd9e74db7ff.jpg" alt="Base module-P5"/>
  This image is provided by RangeKing@GitHub, thanks very much!

## User guide for this documentation

MMYOLO divides the document structure into 6 parts, corresponding to different user needs.

- **Get started with MMYOLO**. This part is must read for first-time MMYOLO users, so please read it carefully.
- **Recommend Topics**. This part is the essence documentation provided in MMYOLO by topics, including lots of MMYOLO features, etc. Highly recommended reading for all MMYOLO users.
- **Common functions**. This part provides a list of common features that you will use during the training and testing process, so you can refer back to them when you need.
- **Useful tools**. This part is useful tools summary under `tools`, so that you can quickly and happily use the various scripts provided in MMYOLO.
- **Basic and advanced tutorials**. This part introduces some basic concepts and advanced tutorials in MMYOLO. It is suitable for users who want to understand the design idea and structure design of MMYOLO in detail.
- **Others**. The rest includes model repositories, specifications and interface documentation, etc.

Users with different needs can choose your favorite content to read. If you have any questions about this documentation or a better idea to improve it, welcome to post a Pull Request to MMYOLO ~. Please refer to [How to Contribute to MMYOLO](../recommended_topics/contributing.md)
