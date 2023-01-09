# Get Started

## Prerequisites

Compatible MMEngine, MMCV and MMDetection versions are shown as below. Please install the correct version to avoid installation issues.

| MMYOLO version |   MMDetection version    |     MMEngine version     |      MMCV version       |
| :------------: | :----------------------: | :----------------------: | :---------------------: |
|      main      | mmdet>=3.0.0rc5, \<3.1.0 | mmengine>=0.3.1, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.3.0      | mmdet>=3.0.0rc5, \<3.1.0 | mmengine>=0.3.1, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.2.0      | mmdet>=3.0.0rc3, \<3.1.0 | mmengine>=0.3.1, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.1.3      | mmdet>=3.0.0rc3, \<3.1.0 | mmengine>=0.3.1, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.1.2      | mmdet>=3.0.0rc2, \<3.1.0 | mmengine>=0.3.0, \<1.0.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.1.1      |     mmdet==3.0.0rc1      | mmengine>=0.1.0, \<0.2.0 | mmcv>=2.0.0rc0, \<2.1.0 |
|     0.1.0      |     mmdet==3.0.0rc0      | mmengine>=0.1.0, \<0.2.0 | mmcv>=2.0.0rc0, \<2.1.0 |

In this section, we demonstrate how to prepare an environment with PyTorch.

MMDetection works on Linux, Windows, and macOS. It requires Python 3.6+, CUDA 9.2+, and PyTorch 1.7+.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

## Installation

### Best Practices

**Step 0.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install "mmengine>=0.3.1"
mim install "mmcv>=2.0.0rc1,<2.1.0"
mim install "mmdet>=3.0.0rc5,<3.1.0"
```

**Note:**

a. In MMCV-v2.x, `mmcv-full` is rename to `mmcv`, if you want to install `mmcv` without CUDA ops, you can use `mim install "mmcv-lite>=2.0.0rc1"` to install the lite version.

b. If you would like to use albumentations, we suggest using pip install -r requirements/albu.txt or pip install -U albumentations --no-binary qudida,albumentations. If you simply use pip install albumentations==1.0.1, it will install opencv-python-headless simultaneously (even though you have already installed opencv-python). We recommended checking the environment after installing albumentation to ensure that opencv-python and opencv-python-headless are not installed at the same time, because it might cause unexpected issues if they both installed. Please refer to [official documentation](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies) for more details.

**Step 1.** Install MMYOLO.

Case a: If you develop and run mmdet directly, install it from source:

```shell
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Case b: If you use MMYOLO as a dependency or third-party package, install it with MIM:

```shell
mim install "mmyolo"
```

## Verify the installation

To verify whether MMYOLO is installed correctly, we provide some sample codes to run an inference demo.

**Step 1.** We need to download config and checkpoint files.

```shell
mim download mmyolo --config yolov5_s-v61_syncbn_fast_8xb16-300e_coco --dest .
```

The downloading will take several seconds or more, depending on your network environment. When it is done, you will find two files `yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py` and `yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth` in your current folder.

**Step 2.** Verify the inference demo.

Option (a). If you install MMYOLO from source, just run the following command.

```shell
python demo/image_demo.py demo/demo.jpg \
                          yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                          yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth

# Optional parameters
# --out-dir ./output *The detection results are output to the specified directory. When args have action --show, the script do not save results. Default: ./output
# --device cuda:0    *The computing resources used, including cuda and cpu. Default: cuda:0
# --show             *Display the results on the screen. Default: False
# --score-thr 0.3    *Confidence threshold. Default: 0.3
```

You will see a new image on your `output` folder, where bounding boxes are plotted.

Supported input types:

- Single image, include `jpg`, `jpeg`, `png`, `ppm`, `bmp`, `pgm`, `tif`, `tiff`, `webp`.
- Folder, all image files in the folder will be traversed and the corresponding results will be output.
- URL, will automatically download from the URL and the corresponding results will be output.

Option (b). If you install MMYOLO with MIM, open your python interpreter and copy&paste the following codes.

```python
from mmdet.apis import init_detector, inference_detector
from mmyolo.utils import register_all_modules

register_all_modules()
config_file = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
checkpoint_file = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```

You will see a list of `DetDataSample`, and the predictions are in the `pred_instance`, indicating the detected bounding boxes, labels, and scores.

### Customize Installation

#### CUDA versions

When installing PyTorch, you need to specify the version of CUDA. If you are not clear on which to choose, follow our recommendations:

- For Ampere-based NVIDIA GPUs, such as GeForce 30 series and NVIDIA A100, CUDA 11 is a must.
- For older NVIDIA GPUs, CUDA 11 is backward compatible, but CUDA 10.2 offers better compatibility and is more lightweight.

Please make sure the GPU driver satisfies the minimum version requirements. See [this table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions) for more information.

```{note}
Installing CUDA runtime libraries is enough if you follow our best practices, because no CUDA code will be compiled locally. However, if you hope to compile MMCV from source or develop other CUDA operators, you need to install the complete CUDA toolkit from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads), and its version should match the CUDA version of PyTorch. i.e., the specified version of cudatoolkit in `conda install` command.
```

#### Install MMEngine without MIM

To install MMEngine with pip instead of MIM, please follow \[MMEngine installation guides\](https://mmengine.readthedocs.io/en/latest/get_started/installation.html).

For example, you can install MMEngine by the following command.

```shell
pip install "mmengine>=0.3.1"
```

#### Install MMCV without MIM

MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way. MIM solves such dependencies automatically and makes the installation easier. However, it is not a must.

To install MMCV with pip instead of MIM, please follow [MMCV installation guides](https://mmcv.readthedocs.io/en/2.x/get_started/installation.html). This requires manually specifying a find-url based on the PyTorch version and its CUDA version.

For example, the following command installs MMCV built for PyTorch 1.12.x and CUDA 11.6.

```shell
pip install "mmcv>=2.0.0rc1" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

#### Install on CPU-only platforms

MMDetection can be built for the CPU-only environment. In CPU mode you can train (requires MMCV version >= `2.0.0rc1`), test, or infer a model.

However, some functionalities are gone in this mode:

- Deformable Convolution
- Modulated Deformable Convolution
- ROI pooling
- Deformable ROI pooling
- CARAFE
- SyncBatchNorm
- CrissCrossAttention
- MaskedConv2d
- Temporal Interlace Shift
- nms_cuda
- sigmoid_focal_loss_cuda
- bbox_overlaps

If you try to train/test/infer a model containing the above ops, an error will be raised.
The following table lists affected algorithms.

|                        Operator                         |                                          Model                                           |
| :-----------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| Deformable Convolution/Modulated Deformable Convolution | DCN、Guided Anchoring、RepPoints、CentripetalNet、VFNet、CascadeRPN、NAS-FCOS、DetectoRS |
|                      MaskedConv2d                       |                                     Guided Anchoring                                     |
|                         CARAFE                          |                                          CARAFE                                          |
|                      SyncBatchNorm                      |                                         ResNeSt                                          |

#### Install on Google Colab

[Google Colab](https://research.google.com/) usually has PyTorch installed,
thus we only need to install MMEngine, MMCV, MMDetection, and MMYOLO with the following commands.

**Step 1.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
!pip3 install openmim
!mim install "mmengine==0.1.0"
!mim install "mmcv>=2.0.0rc1,<2.1.0"
!mim install "mmdet>=3.0.0rc5,<3.1.0"
```

**Step 2.** Install MMYOLO from the source.

```shell
!git clone https://github.com/open-mmlab/mmyolo.git
%cd mmyolo
!pip install -e .
```

**Step 3.** Verification.

```python
import mmyolo
print(mmyolo.__version__)
# Example output: 0.1.0, or an another version.
```

```{note}
Within Jupyter, the exclamation mark `!` is used to call external executables and `%cd` is a [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd) to change the current working directory of Python.
```

#### Using MMYOLO with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmyolo/blob/master/docker/Dockerfile) to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

Reminder: If you find out that your download speed is very slow, we suggest that you can canceling the comments in the last two lines of `Optional` in the [Dockerfile](https://github.com/open-mmlab/mmyolo/blob/master/docker/Dockerfile#L19-L20) to obtain a rocket like download speed:

```dockerfile
# (Optional)
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Build Command：

```shell
# build an image with PyTorch 1.9, CUDA 11.1
# If you prefer other versions, just modified the Dockerfile
docker build -t mmyolo docker/
```

Run it with:

```shell
export DATA_DIR=/path/to/your/dataset
docker run --gpus all --shm-size=8g -it -v ${DATA_DIR}:/mmyolo/data mmyolo
```

### Troubleshooting

If you have some issues during the installation, please first view the [FAQ](notes/faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmyolo/issues/new/choose) on GitHub if no solution is found.

### Develop using multiple MMYOLO versions

The training and testing scripts have been modified in `PYTHONPATH` to ensure that the scripts use MMYOLO in the current directory.

To have the default MMYOLO installed in your environment instead of what is currently in use, you can remove the code that appears in the relevant script:

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
