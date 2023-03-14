# Installation

## Best Practices

**Step 0.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0rc6,<3.1.0"
```

If you are currently in the mmyolo project directory, you can use the following simplified commands

```shell
cd mmyolo
pip install -U openmim
mim install -r requirements/mminstall.txt
```

**Note:**

a. In MMCV-v2.x, `mmcv-full` is rename to `mmcv`, if you want to install `mmcv` without CUDA ops, you can use `mim install "mmcv-lite>=2.0.0rc1"` to install the lite version.

b. If you would like to use `albumentations`, we suggest using `pip install -r requirements/albu.txt` or `pip install -U albumentations --no-binary qudida,albumentations`. If you simply use `pip install albumentations==1.0.1`, it will install `opencv-python-headless` simultaneously (even though you have already installed `opencv-python`). We recommended checking the environment after installing albumentation to ensure that `opencv-python` and `opencv-python-headless` are not installed at the same time, because it might cause unexpected issues if they both installed. Please refer to [official documentation](https://albumentations.ai/docs/getting_started/installation/#note-on-opencv-dependencies) for more details.

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

To verify whether MMYOLO is installed correctly, we provide  an inference demo.

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

config_file = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
checkpoint_file = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```

You will see a list of `DetDataSample`, and the predictions are in the `pred_instance`, indicating the detected bounding boxes, labels, and scores.

## Using MMYOLO with Docker

We provide a [Dockerfile](https://github.com/open-mmlab/mmyolo/blob/main/docker/Dockerfile) to build an image. Ensure that your [docker version](https://docs.docker.com/engine/install/) >=19.03.

Reminder: If you find out that your download speed is very slow, we suggest canceling the comments in the last two lines of `Optional` in the [Dockerfile](https://github.com/open-mmlab/mmyolo/blob/main/docker/Dockerfile#L19-L20) to obtain a rocket like download speed:

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

For other customized inatallation, see [Customized Installation](../tutorials/custom_installation.md)

## Troubleshooting

If you have some issues during the installation, please first view the [FAQ](../tutorials/faq.md) page.
You may [open an issue](https://github.com/open-mmlab/mmyolo/issues/new/choose) on GitHub if no solution is found.
