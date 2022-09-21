# 实用工具

我们在 `tools/` 文件夹下提供很多实用工具。 除此之外，你也可以通过 MIM 来快速运行 OpenMMLab 的其他开源库。

以 MMDetection 为例，如果想利用 [print_config.py](https://github.com/open-mmlab/mmdetection/blob/3.x/tools/misc/print_config.py)，你可以直接采用如下命令，而无需复制源码到 MMYOLO 库中。

```shell
mim run mmdet print_config [CONFIG]
```

**注意**：上述命令能够成功的前提是 MMDetection 库必须通过 MIM 来安装。

## 可视化

### 可视化 COCO 标签

脚本 `tools/analysis_tools/browse_coco_json.py` 能够使用可视化显示 COCO 标签在图片的情况

```shell
python tools/analysis_tools/browse_coco_json.py ${DATA_ROOT} \
                                                [--ann_file ${ANN_FILE}] \
                                                [--img_dir ${IMG_DIR}] \
                                                [--wait-time ${WAIT_TIME}] \
                                                [--disp-all] [--category-names CATEGORY_NAMES [CATEGORY_NAMES ...]] \
                                                [--shuffle]
```

例子：

1. 查看 `COCO` 全部类别，同时展示 `bbox`、`mask` 等所有类型的标注：

```shell
python tools/analysis_tools/browse_coco_json.py './data/coco/' \
                                                --ann_file 'annotations/instances_train2017.json' \
                                                --img_dir 'train2017' \
                                                --disp-all
```

2. 查看 `COCO` 全部类别，同时仅展示 `bbox` 类型的标注，并打乱显示：

```shell
python tools/analysis_tools/browse_coco_json.py './data/coco/' \
                                                --ann_file 'annotations/instances_train2017.json' \
                                                --img_dir 'train2017' \
                                                --shuffle
```

3. 只查看 `bicycle` 和 `person` 类别，同时仅展示 `bbox` 类型的标注：

```shell
python tools/analysis_tools/browse_coco_json.py './data/coco/' \
                                                --ann_file 'annotations/instances_train2017.json' \
                                                --img_dir 'train2017' \
                                                --category-names 'bicycle' 'person'
```

4. 查看 `COCO` 全部类别，同时展示 `bbox`、`mask` 等所有类型的标注，并打乱显示：

```shell
python tools/analysis_tools/browse_coco_json.py './data/coco/' \
                                                --ann_file 'annotations/instances_train2017.json' \
                                                --img_dir 'train2017' \
                                                --disp-all \
                                                --shuffle
```

### 可视化数据集

脚本 `tools/analysis_tools/browse_dataset.py` 能够帮助用户去直接窗口可视化数据集的原始图片+展示标签的图片，或者保存可视化图片到指定文件夹内。

```shell
python tools/analysis_tools/browse_dataset.py ${CONFIG} \
                                              [-h] \
                                              [--output-dir ${OUTPUT_DIR}] \
                                              [--not-show] \
                                              [--show-interval ${SHOW_INTERVAL}]
```

例子：

1. 使用 `config` 文件 `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py` 可视化图片，图片直接弹出显示，同时保存到目录 `work-dir/browse_dataset`：

```shell
python tools/analysis_tools/browse_dataset.py 'configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py' \
                                               --output-dir 'work-dir/browse_dataset'
```

2. 使用 `config` 文件 `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py` 可视化图片，图片直接弹出显示，每张图片持续 `10` 秒，同时保存到目录 `work-dir/browse_dataset`：

```shell
python tools/analysis_tools/browse_dataset.py 'configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py' \
                                               --output-dir 'work-dir/browse_dataset' \
                                               --show-interval 10
```

3. 使用 `config` 文件 `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py` 可视化图片，图片直接弹出显示，每张图片持续 `10` 秒，图片不进行保存：

```shell
python tools/analysis_tools/browse_dataset.py 'configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py' \
                                               --show-interval 10
```

4. 使用 `config` 文件 `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py` 可视化图片，图片不直接弹出显示，仅保存到目录 `work-dir/browse_dataset`：

```shell
python tools/analysis_tools/browse_dataset.py 'configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py' \
                                               --output-dir 'work-dir/browse_dataset' \
                                               --not-show
```

## 数据集转换

文件夹 `tools/data_converters/` 包含工具将 `balloon` 数据集（该小型数据集仅作为入门使用）转换成 COCO 的格式。

关于该脚本的详细说明，请看 [YOLOv5 从入门到部署全流程](./yolov5_tutorial.md) 中 `数据集准备` 小节。

```shell
python tools/dataset_converters/balloon2coco.py
```

## 数据集下载

脚本 `tools/misc/download_dataset.py` 支持下载数据集，例如 `COCO`、`VOC`、`LVIS` 和 `Balloon`.

```shell
python tools/misc/download_dataset.py --dataset-name coco2017
python tools/misc/download_dataset.py --dataset-name voc2007
python tools/misc/download_dataset.py --dataset-name lvis
python tools/misc/download_dataset.py --dataset-name balloon [--save-dir ${SAVE_DIR}] [--unzip]
```

## 模型转换

文件夹 `tools/analysis_tools/` 下的三个脚本能够帮助用户将对应YOLO官方的预训练模型中的键转换成 `MMYOLO` 格式，并使用 `MMYOLO` 对模型进行微调。

### YOLOv5

下面以转换 `yolov5s.pt` 为例：

1. 将 YOLOv5 官方代码克隆到本地（目前支持的最高版本为 `v6.1` ）：

```shell
git clone -b v6.1 https://github.com/ultralytics/yolov5.git
cd yolov5
```

2. 下载官方权重：

```shell
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
```

3. 将 `tools/model_converters/yolov5_to_mmyolo.py` 文件复制到 YOLOv5 官方代码克隆的路径：

```shell
cp ${MMDET_YOLO_PATH}/tools/model_converters/yolov5_to_mmyolo.py yolov5_to_mmyolo.py
```

4. 执行转换：

```shell
python yolov5_to_mmyolo.py --src ${WEIGHT_FILE_PATH} --dst mmyolov5.pt
```

转换好的 `mmyolov5.pt` 即可以为 MMYOLO 所用。 YOLOv6 官方权重转化也是采用一样的使用方式。

### YOLOX

YOLOX 模型的转换不需要下载 YOLOX 官方代码，只需要下载权重即可。下面以转换 `yolox_s.pth` 为例：

1. 下载权重：

```shell
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

2. 执行转换：

```shell
python tools/model_converters/yolox_to_mmyolo.py --src yolox_s.pth --dst mmyolox.pt
```

转换好的 `mmyolox.pt` 即可以在 MMYOLO 中使用。
