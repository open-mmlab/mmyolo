# 依赖

下表为 MMYOLO 和 MMEngine, MMCV, MMDetection 依赖库的版本要求，请安装正确的版本以避免安装问题。

| MMYOLO version |   MMDetection version    |     MMEngine version     |      MMCV version       |
| :------------: | :----------------------: | :----------------------: | :---------------------: |
|      main      | mmdet>=3.0.0rc5, \<3.1.0 | mmengine>=0.3.1, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.3.0      | mmdet>=3.0.0rc5, \<3.1.0 | mmengine>=0.3.1, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.2.0      | mmdet>=3.0.0rc3, \<3.1.0 | mmengine>=0.3.1, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.1.3      | mmdet>=3.0.0rc3, \<3.1.0 | mmengine>=0.3.1, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.1.2      | mmdet>=3.0.0rc2, \<3.1.0 | mmengine>=0.3.0, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.1.1      |     mmdet==3.0.0rc1      | mmengine>=0.1.0, \<0.2.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.1.0      |     mmdet==3.0.0rc0      | mmengine>=0.1.0, \<0.2.0 | mmcv>=2.0.0rc0, \<2.1.0 |

本节中，我们将演示如何用 PyTorch 准备一个环境。

MMYOLO 支持在 Linux，Windows 和 macOS 上运行。它需要 Python 3.7 以上，CUDA 9.2 以上和 PyTorch 1.7 以上。

```{note}
如果你对 PyTorch 有经验并且已经安装了它，你可以直接跳转到[下一小节](#安装流程)。否则，你可以按照下述步骤进行准备
```

**步骤 0.** 从 [官方网站](https://docs.conda.io/en/latest/miniconda.html) 下载并安装 Miniconda。

**步骤 1.** 创建并激活一个 conda 环境。

```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
```

**步骤 2.** 基于 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装 PyTorch。

在 GPU 平台上：

```shell
conda install pytorch torchvision -c pytorch
```

在 CPU 平台上:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```
