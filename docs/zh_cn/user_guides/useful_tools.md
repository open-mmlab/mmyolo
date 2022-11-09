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

脚本 `tools/analysis_tools/dataset_analysis.py` 能够帮助用户得到四种功能的结果图，并将图片保存到当前运行目录下的 `dataset_analysis` 文件夹中。
关于该脚本的功能的说明：
通过 `main()` 的数据准备，得到每个子函数所需要的数据。
功能一：显示类别和 bbox 实例个数的分布图，通过子函数 `show_bbox_num` 生成。

<img src="https://user-images.githubusercontent.com/90811472/200314770-4fb21626-72f2-4a4c-be5d-bf860ad830ec.jpg"/>

功能二：显示类别和 bbox 实例宽、高的分布图，通过子函数 `show_bbox_wh` 生成。

<img src="https://user-images.githubusercontent.com/90811472/200315007-96e8e795-992a-4c72-90fa-f6bc00b3f2c7.jpg"/>

功能三：显示类别和 bbox 实例宽/高比例的分布图，通过子函数 `show_bbox_wh_ratio` 生成。

<img src="https://user-images.githubusercontent.com/90811472/200315044-4bdedcf6-087a-418e-8fe8-c2d3240ceba8.jpg"/>

功能四：基于面积规则下，显示类别和 bbox 实例面积的分布图，通过子函数 `show_bbox_area` 生成。

<img src="https://user-images.githubusercontent.com/90811472/200315075-71680fe2-db6f-4981-963e-a035c1281fc1.jpg"/>

打印列表显示，通过脚本中子函数 `show_class_list` 和 `show_data_list` 生成。

<img src="https://user-images.githubusercontent.com/90811472/200315152-9d6df91c-f2d2-4bba-9f95-b790fac37b62.jpg"/>

```shell
python tools/analysis_tools/dataset_analysis.py ${CONFIG} \
                                              [-h] \
                                              [--val-dataset ${TYPE}] \
                                              [--class-name ${CLASS_NAME}] \
                                              [--area-rule ${AREA_RULE}] \
                                              [--func ${FUNC}] \
                                              [--output-dir ${OUTPUT_DIR}]
```

例子：

1.使用 `config` 文件 `configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py` 分析数据集，其中默认设置:数据加载类型为 `train_dataset` ，面积规则设置为 `[0,32,96,1e5]` ,生成包含所有类的结果图并将图片保存到当前运行目录下 `./dataset_analysis` 文件夹中：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py
```

2.使用 `config` 文件 `configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py` 分析数据集，通过 `--val-dataset` 设置将数据加载类型由默认的 `train_dataset` 改为 `val_dataset`：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py \
                                               --val-dataset
```

3.使用 `config` 文件 `configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py` 分析数据集，通过 `--class-name` 设置将生成所有类改为特定类显示，以显示 `person` 为例：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py \
                                               --class-name person
```

4.使用 `config` 文件 `configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py` 分析数据集，通过 `--area-rule` 重新定义面积规则，以 `30 70 125` 为例,面积规则变为 `[0,30,70,125,1e5]`：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py \
                                               --area-rule 30 70 125
```

5.使用 `config` 文件 `configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py` 分析数据集，通过 `--func` 设置，将显示四个功能效果图改为只显示 `功能一` 为例：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py \
                                               --func show_bbox_num
```

6.使用 `config` 文件 `configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py` 分析数据集，通过 `--output-dir` 设置修改图片保存地址，以 `work_ir/dataset_analysis` 地址为例：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py \
                                               --output-dir work_dir/dataset_analysis
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

## 优化锚框尺寸

脚本 `tools/analysis_tools/optimize_anchors.py` 支持 yolo 系列中三种锚框生成方式，分别是 `k-means`、`differential_evolution`、`v5-k-means`.

### k-means

在 k-means 方法中，使用的是基于 IoU 表示距离的聚类方法，具体使用命令如下:

```shell
python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
    --algorithm k-means \
    --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
    --output-dir ${OUTPUT_DIR}
```

### differential_evolution

在 differential_evolution 方法中，使用的是基于差分进化算法（简称 DE 算法）的聚类方式，其最小化目标函数为 `avg_iou_cost` ，具体使用命令如下:

```shell
python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
    --algorithm differential_evolution \
    --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
    --output-dir ${OUTPUT_DIR}
```

### v5-k-means

在 v5-k-means 方法中，使用的是 yolov5 中基于 shape-match 的聚类方式，具体使用命令如下:

```shell
python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
    --algorithm v5-k-means \
    --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
    --prior_match_thr ${PRIOR_MATCH_THR} \
    --output-dir ${OUTPUT_DIR}
```

## 提取 COCO 子集

COCO2017 数据集训练数据集包括 118K 张图片，验证集包括 5K 张图片，数据集比较大。在调试或者快速验证程序是否正确的场景下加载 json 会需要消耗较多资源和带来较慢的启动速度，这会导致程序体验不好。
`extract_subcoco.py` 脚本提供了切分指定张图片的功能，用户可以通过 `--num-img` 参数来得到指定图片数目的 COCO 子集，从而满足上述需求。

注意： 本脚本目前仅仅支持 COCO2017 数据集，未来会支持更加通用的 COCO JSON 格式数据集

输入 root 根路径文件夹格式如下所示：

```text
├── root
│   ├── annotations
│   ├── train2017
│   ├── val2017
│   ├── test2017
```

1. 仅仅使用 5K 张验证集切分出 10 张训练图片和 10 张验证图片

```shell
python tools/misc/extract_subcoco.py ${ROOT} ${OUT_DIR} --num-img 10
```

2. 使用训练集切分出 20 张训练图片，使用验证集切分出 20 张验证图片

```shell
python tools/misc/extract_subcoco.py ${ROOT} ${OUT_DIR} --num-img 20 --use-training-set
```

3. 设置全局种子，默认不设置

```shell
python tools/misc/extract_subcoco.py ${ROOT} ${OUT_DIR} --num-img 20 --use-training-set --seed 1
```
