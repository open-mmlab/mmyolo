# Prerequisites

Compatible MMEngine, MMCV and MMDetection versions are shown as below. Please install the correct version to avoid installation issues.

| MMYOLO version |   MMDetection version    |     MMEngine version     |      MMCV version       |
| :------------: | :----------------------: | :----------------------: | :---------------------: |
|      main      |  mmdet>=3.0.0, \<3.1.0   | mmengine>=0.7.1, \<1.0.0 | mmcv>=2.0.0rc4, \<2.1.0 |
|     0.6.0      |  mmdet>=3.0.0, \<3.1.0   | mmengine>=0.7.1, \<1.0.0 | mmcv>=2.0.0rc4, \<2.1.0 |
|     0.5.0      | mmdet>=3.0.0rc6, \<3.1.0 | mmengine>=0.6.0, \<1.0.0 | mmcv>=2.0.0rc4, \<2.1.0 |
|     0.4.0      | mmdet>=3.0.0rc5, \<3.1.0 | mmengine>=0.3.1, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.3.0      | mmdet>=3.0.0rc5, \<3.1.0 | mmengine>=0.3.1, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.2.0      | mmdet>=3.0.0rc3, \<3.1.0 | mmengine>=0.3.1, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.1.3      | mmdet>=3.0.0rc3, \<3.1.0 | mmengine>=0.3.1, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.1.2      | mmdet>=3.0.0rc2, \<3.1.0 | mmengine>=0.3.0, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.1.1      |     mmdet==3.0.0rc1      | mmengine>=0.1.0, \<0.2.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.1.0      |     mmdet==3.0.0rc0      | mmengine>=0.1.0, \<0.2.0 | mmcv>=2.0.0rc0, \<2.1.0 |

In this section, we demonstrate how to prepare an environment with PyTorch.

MMDetection works on Linux, Windows, and macOS. It requires:

- Python 3.7+
- PyTorch 1.7+
- CUDA 9.2+
- GCC 5.4+

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official commands](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

**Step 3.** Verify PyTorch installation

```shell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If the GPU is used, the version information and `True` are printed; otherwise, the version information and `False` are printed.
