# 中文解读资源汇总

本文汇总了 MMYOLO 或相关的 [OpenMMLab](https://www.zhihu.com/people/openmmlab) 解读的部分文章（更多文章和视频见 [OpenMMLabCourse](https://github.com/open-mmlab/OpenMMLabCourse) )，如果您有推荐的文章（不一定是 OpenMMLab 发布的文章，可以是自己写的文章），非常欢迎提 Pull Request 添加到这里。

## MMYOLO 解读文章和资源

### 脚本命令速查表

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/213104312-3580c783-2423-442f-b5f6-79204a06adb5.png">
</div>

你可以点击[链接](https://pan.baidu.com/s/1QEaqT7YayUdEvh1an0gjHg?pwd=yolo)，下载高清版 PDF 文件。

### 文章

- [社区协作，简洁易用，快来开箱新一代 YOLO 系列开源库](https://zhuanlan.zhihu.com/p/575615805)
- [MMYOLO 社区倾情贡献，RTMDet 原理社区开发者解读来啦！](https://zhuanlan.zhihu.com/p/569777684)
- [MMYOLO 自定义数据集从标注到部署保姆级教程](https://zhuanlan.zhihu.com/p/595497726)
- [满足一切需求的 MMYOLO 可视化：测试过程可视化](https://zhuanlan.zhihu.com/p/593179372)
- [MMYOLO 想你所想: 训练过程可视化](https://zhuanlan.zhihu.com/p/608586878)
- [YOLOv8 深度详解！一文看懂，快速上手](https://zhuanlan.zhihu.com/p/598566644)
- [玩转 MMYOLO 基础类第一期： 配置文件太复杂？继承用法看不懂？配置全解读来了](https://zhuanlan.zhihu.com/p/577715188)
- [玩转 MMYOLO 工具类第一期： 特征图可视化](https://zhuanlan.zhihu.com/p/578141381?)
- [玩转 MMYOLO 实用类第一期：源码阅读和调试「必备」技巧文档](https://zhuanlan.zhihu.com/p/580885852)
- [玩转 MMYOLO 基础类第二期：工程文件结构简析](https://zhuanlan.zhihu.com/p/584807195)
- [玩转 MMYOLO 实用类第二期：10分钟换遍主干网络文档](https://zhuanlan.zhihu.com/p/585641598)

### 视频

#### 工具类

|       |         内容         |                                                                                                                                                                                                      视频                                                                                                                                                                                                       |                                                                                                                                      课程中的代码/文档                                                                                                                                      |
| :---: | :------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| 第1讲 |     特征图可视化     |  [![Link](https://i2.hdslb.com/bfs/archive/480a0eb41fce26e0acb65f82a74501418eee1032.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV188411s7o8)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV188411s7o8)](https://www.bilibili.com/video/BV188411s7o8)  | [特征图可视化文档](https://zhuanlan.zhihu.com/p/578141381)<br>[特征图可视化.ipynb](https://github.com/open-mmlab/OpenMMLabCourse/blob/main/codes/MMYOLO_tutorials/%5B%E5%B7%A5%E5%85%B7%E7%B1%BB%E7%AC%AC%E4%B8%80%E6%9C%9F%5D%E7%89%B9%E5%BE%81%E5%9B%BE%E5%8F%AF%E8%A7%86%E5%8C%96.ipynb) |
| 第2讲 | 基于 sahi 的大图推理 | [![Link](https://i0.hdslb.com/bfs/archive/62c41f508dbcf63a4c721738171612d2d7069ac2.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1EK411R7Ws/)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1EK411R7Ws)](https://www.bilibili.com/video/BV1EK411R7Ws/) |                                                                  [10分钟轻松掌握大图推理.ipynb](https://github.com/open-mmlab/OpenMMLabCourse/blob/main/codes/MMYOLO_tutorials/[工具类第二期]10分钟轻松掌握大图推理.ipynb)                                                                  |

#### 基础类

|       |       内容       |                                                                                                                                                                                                     视频                                                                                                                                                                                                      |                       课程中的代码/文档                        |
| :---: | :--------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------: |
| 第1讲 |    配置全解读    | [![Link](https://i1.hdslb.com/bfs/archive/e06daf640ea39b3c0700bb4dc758f1a253f33e13.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1214y157ck)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1214y157ck)](https://www.bilibili.com/video/BV1214y157ck) |    [配置全解读文档](https://zhuanlan.zhihu.com/p/577715188)    |
| 第2讲 | 工程文件结构简析 |  [![Link](https://i2.hdslb.com/bfs/archive/41030efb84d0cada06d5451c1e6e9bccc0cdb5a3.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1LP4y117jS)[![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1LP4y117jS)](https://www.bilibili.com/video/BV1LP4y117jS)  | [工程文件结构简析文档](https://zhuanlan.zhihu.com/p/584807195) |

#### 实用类

|       |                内容                |                                                                                                                                                                                                     视频                                                                                                                                                                                                      |                                                                                                   课程中的代码/文档                                                                                                   |
| :---: | :--------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| 第1讲 |     源码阅读和调试「必备」技巧     | [![Link](https://i2.hdslb.com/bfs/archive/790d2422c879ff20488910da1c4422b667ea6af7.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1N14y1V7mB)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1N14y1V7mB)](https://www.bilibili.com/video/BV1N14y1V7mB) |                                                                       [源码阅读和调试「必备」技巧文档](https://zhuanlan.zhihu.com/p/580885852)                                                                        |
| 第2讲 |         10分钟换遍主干网络         | [![Link](https://i0.hdslb.com/bfs/archive/c51f1aef7c605856777249a7b4478f44bd69f3bd.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1JG4y1d7GC)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1JG4y1d7GC)](https://www.bilibili.com/video/BV1JG4y1d7GC) | [10分钟换遍主干网络文档](https://zhuanlan.zhihu.com/p/585641598)<br>[10分钟换遍主干网络.ipynb](https://github.com/open-mmlab/OpenMMLabCourse/blob/main/codes/MMYOLO_tutorials/[实用类第二期]10分钟换遍主干网络.ipynb) |
| 第3讲 | 自定义数据集从标注到部署保姆级教程 | [![Link](https://i2.hdslb.com/bfs/archive/13f566c89a18c9c881713b63ec14da952d4c0b14.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1RG4y137i5)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1RG4y137i5)](https://www.bilibili.com/video/BV1RG4y137i5) |                                                            [自定义数据集从标注到部署保姆级教程](../recommended_topics/labeling_to_deployment_tutorials.md)                                                            |
| 第4讲 |      顶会第一步 · 模块自定义       | [![Link](http://i2.hdslb.com/bfs/archive/5b23d41ac57466824eaf185ef806ef734414e93b.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1yd4y1j7VD)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1yd4y1j7VD)](https://www.bilibili.com/video/BV1yd4y1j7VD)  |                                [顶会第一步·模块自定义.ipynb](https://github.com/open-mmlab/OpenMMLabCourse/blob/main/codes/MMYOLO_tutorials/[实用类第四期]顶会第一步·模块自定义.ipynb)                                |

#### 源码解读类

#### 演示类

|       |     内容     |                                                                                                                                                                                                      视频                                                                                                                                                                                                       |
| :---: | :----------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| 第1期 | 特征图可视化 | [![Link](https://i0.hdslb.com/bfs/archive/081f300c84d6556f40d984cfbe801fc0644ff449.jpg@112w_63h_1c.webp)](https://www.bilibili.com/video/BV1je4y1478R/)  [![bilibili](https://img.shields.io/badge/dynamic/json?label=views&style=social&logo=bilibili&query=data.stat.view&url=https%3A%2F%2Fapi.bilibili.com%2Fx%2Fweb-interface%2Fview%3Fbvid%3DBV1je4y1478R)](https://www.bilibili.com/video/BV1je4y1478R/) |

## MMDetection 解读文章和资源

### 文章

- [MMDetection 3.0：目标检测新基准与前沿](https://zhuanlan.zhihu.com/p/575246786)
- [目标检测、实例分割、旋转框样样精通！详解高性能检测算法 RTMDet](https://zhuanlan.zhihu.com/p/598846422)
- [MMDetection 支持数据增强神器 Simple Copy Paste 全过程](https://zhuanlan.zhihu.com/p/559940982)

### 知乎问答和资源

- [深度学习科研，如何高效进行代码和实验管理？](https://www.zhihu.com/question/269707221/answer/2480772257)
- [深度学习方面的科研工作中的实验代码有什么规范和写作技巧？如何妥善管理实验数据？](https://www.zhihu.com/question/268193800/answer/2586000037)
- [COCO 数据集上 1x 模式下为什么不采用多尺度训练?](https://www.zhihu.com/question/462170786/answer/1915119662)
- [MMDetection 中 SOTA 论文源码中将训练过程中 BN 层的 eval 打开?](https://www.zhihu.com/question/471189603/answer/2195540892)
- [基于 PyTorch 的 MMDetection 中训练的随机性来自何处？](https://www.zhihu.com/question/453511684/answer/1839683634)

## MMEngine 解读文章和资源

- [从 MMCV 到 MMEngine，架构升级，体验升级！](https://zhuanlan.zhihu.com/p/571830155)

## MMCV 解读文章和资源

- [MMCV 全新升级，新增超全数据变换功能，还有两大变化](https://zhuanlan.zhihu.com/p/572550592)
- [手把手教你如何高效地在 MMCV 中贡献算子](https://zhuanlan.zhihu.com/p/464492627)

## PyTorch 解读文章和资源

- [PyTorch1.11 亮点一览：TorchData、functorch、DDP 静态图](https://zhuanlan.zhihu.com/p/486222256)
- [PyTorch1.12 亮点一览：DataPipe + TorchArrow 新的数据加载与处理范式](https://zhuanlan.zhihu.com/p/537868554)
- [PyTorch 源码解读之 nn.Module：核心网络模块接口详解](https://zhuanlan.zhihu.com/p/340453841)
- [PyTorch 源码解读之 torch.autograd：梯度计算详解](https://zhuanlan.zhihu.com/p/321449610)
- [PyTorch 源码解读之 torch.utils.data：解析数据处理全流程](https://zhuanlan.zhihu.com/p/337850513)
- [PyTorch 源码解读之 torch.optim：优化算法接口详解](https://zhuanlan.zhihu.com/p/346205754)
- [PyTorch 源码解读之 DP & DDP：模型并行和分布式训练解析](https://zhuanlan.zhihu.com/p/343951042)
- [PyTorch 源码解读之 BN & SyncBN：BN 与 多卡同步 BN 详解](https://zhuanlan.zhihu.com/p/337732517)
- [PyTorch 源码解读之 torch.cuda.amp: 自动混合精度详解](https://zhuanlan.zhihu.com/p/348554267)
- [PyTorch 源码解读之 cpp_extension：揭秘 C++/CUDA 算子实现和调用全流程](https://zhuanlan.zhihu.com/p/348555597)
- [PyTorch 源码解读之即时编译篇](https://zhuanlan.zhihu.com/p/361101354)
- [PyTorch 源码解读之分布式训练了解一下？](https://zhuanlan.zhihu.com/p/361314953)
- [PyTorch 源码解读之 torch.serialization & torch.hub](https://zhuanlan.zhihu.com/p/364239544)

## 其他

- [Type Hints 入门教程，让代码更加规范整洁](https://zhuanlan.zhihu.com/p/519335398)
