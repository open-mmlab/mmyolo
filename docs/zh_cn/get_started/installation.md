# 安装和验证

## 最佳实践

**步骤 0.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine)、 [MMCV](https://github.com/open-mmlab/mmcv) 和 [MMDetection](https://github.com/open-mmlab/mmdetection) 。

```shell
pip install -U openmim
mim install "mmengine>=0.3.1"
mim install "mmcv>=2.0.0rc1,<2.1.0"
mim install "mmdet>=3.0.0rc5,<3.1.0"
```

**注意：**

a. 在 MMCV-v2.x 中，`mmcv-full` 改名为 `mmcv`，如果你想安装不包含 CUDA 算子精简版，可以通过 `mim install mmcv-lite>=2.0.0rc1` 来安装。

b. 如果使用 albumentations，我们建议使用 pip install -r requirements/albu.txt 或者 pip install -U albumentations --no-binary qudida,albumentations 进行安装。 如果简单地使用 pip install albumentations==1.0.1 进行安装，则会同时安装 opencv-python-headless（即便已经安装了 opencv-python 也会再次安装）。我们建议在安装 albumentations 后检查环境，以确保没有同时安装 opencv-python 和 opencv-python-headless，因为同时安装可能会导致一些问题。更多细节请参考 [官方文档](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies) 。

**步骤 1.** 安装 MMYOLO

方案 1. 如果你基于 MMYOLO 框架开发自己的任务，建议从源码安装

```shell
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

方案 2. 如果你将 MMYOLO 作为依赖或第三方 Python 包，使用 MIM 安装

```shell
mim install "mmyolo"
```

## 验证安装

为了验证 MMYOLO 是否安装正确，我们提供了一些示例代码来执行模型推理。

**步骤 1.** 我们需要下载配置文件和模型权重文件。

```shell
mim download mmyolo --config yolov5_s-v61_syncbn_fast_8xb16-300e_coco --dest .
```

下载将需要几秒钟或更长时间，这取决于你的网络环境。完成后，你会在当前文件夹中发现两个文件 `yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py` and `yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth`。

**步骤 2.** 推理验证

方案 1. 如果你通过源码安装的 MMYOLO，那么直接运行以下命令进行验证：

```shell
python demo/image_demo.py demo/demo.jpg \
                          yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                          yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth

# 可选参数
# --out-dir ./output *检测结果输出到指定目录下，默认为./output, 当--show参数存在时，不保存检测结果
# --device cuda:0    *使用的计算资源，包括cuda, cpu等，默认为cuda:0
# --show             *使用该参数表示在屏幕上显示检测结果，默认为False
# --score-thr 0.3    *置信度阈值，默认为0.3
```

运行结束后，在 `output` 文件夹中可以看到检测结果图像，图像中包含有网络预测的检测框。

支持输入类型包括

- 单张图片, 支持 `jpg`, `jpeg`, `png`, `ppm`, `bmp`, `pgm`, `tif`, `tiff`, `webp`。
- 文件目录，会遍历文件目录下所有图片文件，并输出对应结果。
- 网址，会自动从对应网址下载图片，并输出结果。

方案 2. 如果你通过 MIM 安装的 MMYOLO， 那么可以打开你的 Python 解析器，复制并粘贴以下代码：

```python
from mmdet.apis import init_detector, inference_detector
from mmyolo.utils import register_all_modules

register_all_modules()
config_file = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
checkpoint_file = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```

你将会看到一个包含 `DetDataSample` 的列表，预测结果在 `pred_instance` 里，包含有预测框、预测分数 和 预测类别。

## 自定义安装

### CUDA 版本

在安装 PyTorch 时，你需要指定 CUDA 的版本。如果你不清楚应该选择哪一个，请遵循我们的建议。

- 对于 Ampere 架构的 NVIDIA GPU，例如 GeForce 30 系列 以及 NVIDIA A100，CUDA 11 是必需的。
- 对于更早的 NVIDIA GPU，CUDA 11 是向后兼容 (backward compatible) 的，但 CUDA 10.2 能够提供更好的兼容性，也更加轻量。

请确保你的 GPU 驱动版本满足最低的版本需求，参阅 NVIDIA 官方的 [CUDA 工具箱和相应的驱动版本关系表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)。

```{note}
如果按照我们的最佳实践进行安装，CUDA 运行时库就足够了，因为我们提供相关 CUDA 代码的预编译，不需要进行本地编译。
但如果你希望从源码进行 MMCV 的编译，或是进行其他 CUDA 算子的开发，那么就必须安装完整的 CUDA 工具链，参见
[NVIDIA 官网](https://developer.nvidia.com/cuda-downloads) ，另外还需要确保该 CUDA 工具链的版本与 PyTorch 安装时
的配置相匹配（如用 `conda install` 安装 PyTorch 时指定的 cudatoolkit 版本）。
```

### 不使用 MIM 安装 MMEngine

要使用 pip 而不是 MIM 来安装 MMEngine，请遵照 [MMEngine 安装指南](https://mmengine.readthedocs.io/en/latest/get_started/installation.html)。

例如，你可以通过以下命令安装 MMEngine：

```shell
pip install "mmengine>=0.3.1"
```

### 不使用 MIM 安装 MMCV

MMCV 包含 C++ 和 CUDA 扩展，因此其对 PyTorch 的依赖比较复杂。MIM 会自动解析这些 依赖，选择合适的 MMCV 预编译包，使安装更简单，但它并不是必需的。

要使用 pip 而不是 MIM 来安装 MMCV，请遵照 [MMCV 安装指南](https://mmcv.readthedocs.io/zh_CN/2.x/get_started/installation.html)。
它需要您用指定 URL 的形式手动指定对应的 PyTorch 和 CUDA 版本。

例如，下述命令将会安装基于 PyTorch 1.12.x 和 CUDA 11.6 编译的 mmcv：

```shell
pip install "mmcv>=2.0.0rc1" -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

### 在 CPU 环境中安装

我们的代码能够建立在只使用 CPU 的环境（CUDA 不可用）。

在 CPU 模式下，可以进行模型训练（需要 MMCV 版本 >= `2.0.0rc1`)、测试或者推理，然而以下功能将在 CPU 模式下不能使用：

- Deformable Convolution
- Modulated Deformable Convolution
- ROI pooling
- Deformable ROI pooling
- CARAFE: Content-Aware ReAssembly of FEatures
- SyncBatchNorm
- CrissCrossAttention: Criss-Cross Attention
- MaskedConv2d
- Temporal Interlace Shift
- nms_cuda
- sigmoid_focal_loss_cuda
- bbox_overlaps

因此，如果尝试使用包含上述操作的模型进行训练/测试/推理，将会报错。下表列出了由于依赖上述算子而无法在 CPU 上运行的相关模型：

|                          操作                           |                                           模型                                           |
| :-----------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| Deformable Convolution/Modulated Deformable Convolution | DCN、Guided Anchoring、RepPoints、CentripetalNet、VFNet、CascadeRPN、NAS-FCOS、DetectoRS |
|                      MaskedConv2d                       |                                     Guided Anchoring                                     |
|                         CARAFE                          |                                          CARAFE                                          |
|                      SyncBatchNorm                      |                                         ResNeSt                                          |

### 在 Google Colab 中安装

[Google Colab](https://colab.research.google.com/) 通常已经包含了 PyTorch 环境，因此我们只需要安装 MMEngine、MMCV、MMDetection 和 MMYOLO 即可，命令如下：

**步骤 1.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine) 、 [MMCV](https://github.com/open-mmlab/mmcv) 和 [MMDetection](https://github.com/open-mmlab/mmdetection) 。

```shell
!pip3 install openmim
!mim install "mmengine>=0.3.1"
!mim install "mmcv>=2.0.0rc1,<2.1.0"
!mim install "mmdet>=3.0.0rc5,<3.1.0"
```

**步骤 2.** 使用源码安装 MMYOLO：

```shell
!git clone https://github.com/open-mmlab/mmyolo.git
%cd mmyolo
!pip install -e .
```

**步骤 3.** 验证安装是否成功：

```python
import mmyolo
print(mmyolo.__version__)
# 预期输出: 0.1.0 或其他版本号
```

```{note}
在 Jupyter 中，感叹号 `!` 用于执行外部命令，而 `%cd` 是一个[魔术命令](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd)，用于切换 Python 的工作路径。
```

### 通过 Docker 使用 MMYOLO

我们提供了一个 [Dockerfile](https://github.com/open-mmlab/mmyolo/blob/main/docker/Dockerfile) 来构建一个镜像。请确保你的 [docker 版本](https://docs.docker.com/engine/install/) >=`19.03`。

温馨提示；国内用户建议取消掉 [Dockerfile](https://github.com/open-mmlab/mmyolo/blob/main/docker/Dockerfile#L19-L20) 里面 `Optional` 后两行的注释，可以获得火箭一般的下载提速：

```dockerfile
# (Optional)
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

构建命令：

```shell
# build an image with PyTorch 1.9, CUDA 11.1
# If you prefer other versions, just modified the Dockerfile
docker build -t mmyolo docker/
```

用以下命令运行 Docker 镜像：

```shell
export DATA_DIR=/path/to/your/dataset
docker run --gpus all --shm-size=8g -it -v ${DATA_DIR}:/mmyolo/data mmyolo
```

## 排除故障

如果你在安装过程中遇到一些问题，请先查看 [FAQ](notes/faq.md) 页面。

如果没有找到解决方案，你也可以在 GitHub 上 [打开一个问题](https://github.com/open-mmlab/mmyolo/issues/new/choose)。

## 使用多个 MMYOLO 版本进行开发

训练和测试的脚本已经在 `PYTHONPATH` 中进行了修改，以确保脚本使用当前目录中的 MMYOLO。

要使环境中安装默认的 MMYOLO 而不是当前正在在使用的，可以删除出现在相关脚本中的代码：

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
```
