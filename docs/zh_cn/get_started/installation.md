# 安装和验证

## 最佳实践

**步骤 0.** 使用 [MIM](https://github.com/open-mmlab/mim) 安装 [MMEngine](https://github.com/open-mmlab/mmengine)、 [MMCV](https://github.com/open-mmlab/mmcv) 和 [MMDetection](https://github.com/open-mmlab/mmdetection) 。

```shell
pip install -U openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0,<4.0.0"
```

如果你当前已经处于 mmyolo 工程目录下，则可以采用如下简化写法

```shell
cd mmyolo
pip install -U openmim
mim install -r requirements/mminstall.txt
```

**注意：**

a. 在 MMCV-v2.x 中，`mmcv-full` 改名为 `mmcv`，如果你想安装不包含 CUDA 算子精简版，可以通过 `mim install mmcv-lite>=2.0.0rc1` 来安装。

b. 如果使用 `albumentations`，我们建议使用 `pip install -r requirements/albu.txt` 或者 `pip install -U albumentations --no-binary qudida,albumentations` 进行安装。 如果简单地使用 `pip install albumentations==1.0.1` 进行安装，则会同时安装 `opencv-python-headless`（即便已经安装了 `opencv-python` 也会再次安装）。我们建议在安装 albumentations 后检查环境，以确保没有同时安装 `opencv-python` 和 `opencv-python-headless`，因为同时安装可能会导致一些问题。更多细节请参考 [官方文档](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies) 。

**步骤 1.** 安装 MMYOLO

方案 1. 如果你基于 MMYOLO 框架开发自己的任务，建议从源码安装

```shell
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
mim install -r requirements/albu.txt
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

下载将需要几秒钟或更长时间，这取决于你的网络环境。完成后，你会在当前文件夹中发现两个文件 `yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py` 和 `yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth`。

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

config_file = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
checkpoint_file = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```

你将会看到一个包含 `DetDataSample` 的列表，预测结果在 `pred_instance` 里，包含有预测框、预测分数 和 预测类别。

## 通过 Docker 使用 MMYOLO

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

其余自定义安装流程请查看 [自定义安装](../tutorials/custom_installation.md)

## 排除故障

如果你在安装过程中遇到一些问题，你可以在 GitHub 上 [打开一个问题](https://github.com/open-mmlab/mmyolo/issues/new/choose)。
