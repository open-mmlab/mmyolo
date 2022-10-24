# 实用工具

我们在 `tools/` 文件夹下提供很多实用工具。 除此之外，你也可以通过 MIM 来快速运行 OpenMMLab 的其他开源库。

以 MMDetection 为例，如果想利用 [print_config.py](https://github.com/open-mmlab/mmdetection/blob/3.x/tools/misc/print_config.py)，你可以直接采用如下命令，而无需复制源码到 MMYOLO 库中。

```shell
mim run mmdet print_config [CONFIG]
```

**注意**：上述命令能够成功的前提是 MMDetection 库必须通过 MIM 来安装。

## 可视化

### 可视化 COCO 标签

脚本 `tools/analysis_tools/browse_coco_json.py` 能够使用可视化显示 COCO 标签在图片的情况。

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

### 可视化数据集分析

脚本 `tools/analysis_tools/dataset_analysis.py` 能够帮助用户得到四种功能的效果图，并直接保存可视化图片到指定文件夹内。在指定文件夹内还会生成四个功能对应的文件夹，便于大家对应使用。
关于该脚本的功能的说明：
功能一：显示类别和bbox实例个数的分布图

<div align=center>
< img src="https://user-images.githubusercontent.com/90811472/196891728-4c2f1ab3-01cb-445f-a6b8-39752387c40f.jpg"/>
</div>
功能二：显示类别和bbox实例宽、高的分布图
<div align=center>
< img src="https://user-images.githubusercontent.com/90811472/196891895-ae2bc906-fd63-4896-9c3c-a819c3ce24a6.jpg"/>
</div>
功能三：显示类别和bbox实例宽/高比例的分布图
<div align=center>
< img src="https://user-images.githubusercontent.com/90811472/197392631-8788b4d0-951b-4922-a459-265e055c0ed9.jpg"/>
</div>
功能四：基于物体大、中、小规则下，显示类别和bbox实例面积的分布图
<div align=center>
< img src="https://user-images.githubusercontent.com/90811472/196891947-42a972fc-5bdb-486e-ace9-f5ded4419783.jpg"/>
</div>

其中，生成四个对应文件夹的说明， `show_bbox_num` 保存功能一所实现的图片， `show_bbox_wh` 保存功能二所实现的图片， `show_bbox_wh_ratio` 保存功能三所实现的图片， `show_bbox_area` 保存功能四所实现的图片。

```shell
python tools/analysis_tools/dataset_analysis.py ${CONFIG} \
                                              [--output-dir ${OUTPUT_DIR}]
```

例子：

1.使用 `config` 文件 `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py` 可视化数据集分析，同时得到的结果图片直接保存到当前运行目录中：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py
```

2.使用 `config` 文件 `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py` 可视化数据集分析，同时得到的结果图片直接保存到目录 `work-dir/dataset_analysis`：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py \
                                               --output-dir work-dir/dataset_analysis
```

## 数据集转换

文件夹 `tools/data_converters/` 目前包含 `ballon2coco.py` 和 `yolo2coco.py` 两个数据集转换工具。

- `ballon2coco.py` 将 `balloon` 数据集（该小型数据集仅作为入门使用）转换成 COCO 的格式。

关于该脚本的详细说明，请看 [YOLOv5 从入门到部署全流程](./yolov5_tutorial.md) 中 `数据集准备` 小节。

```shell
python tools/dataset_converters/balloon2coco.py
```

- `yolo2coco.py` 将 `yolo-style` **.txt** 格式的数据集转换成 COCO 的格式，请按如下方式使用：

```shell
python tools/dataset_converters/yolo2coco.py /path/to/the/root/dir/of/your_dataset
```

使用说明：

1. `image_dir` 是需要你传入的待转换的 yolo 格式数据集的根目录，内应包含 `images` 、 `labels` 和 `classes.txt` 文件， `classes.txt` 是当前 dataset 对应的类的声明，一行一个类别。
   `image_dir` 结构如下例所示：

```bash
.
└── $ROOT_PATH
    ├── classes.txt
    ├── labels
    │    ├── a.txt
    │    ├── b.txt
    │    └── ...
    ├── images
    │    ├── a.jpg
    │    ├── b.png
    │    └── ...
    └── ...
```

2. 脚本会检测 `image_dir` 下是否已有的 `train.txt` 、 `val.txt` 和 `test.txt` 。若检测到文件，则会按照类别进行整理， 否则默认不需要分类。故请确保对应的 `train.txt` 、 `val.txt` 和 `test.txt` 要在 `image_dir` 内。文件内的图片路径必须是**绝对路径**。
3. 脚本会默认在 `image_dir` 目录下创建 `annotations` 文件夹并将转换结果存在这里。如果在 `image_dir` 下没找到分类文件，输出文件即为一个 `result.json`，反之则会生成需要的 `train.json` 、 `val.json`、 `test.json`，脚本完成后 `annotations` 结构可如下例所示：

```bash
.
└── $ROOT_PATH
    ├── annotations
    │    ├── result.json
    │    └── ...
    ├── classes.txt
    ├── labels
    │    ├── a.txt
    │    ├── b.txt
    │    └── ...
    ├── images
    │    ├── a.jpg
    │    ├── b.png
    │    └── ...
    └── ...
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
