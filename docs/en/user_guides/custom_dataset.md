# The whole process of custom dataset annotation+training+testing+deployment

In our daily work and study, we often encounter some tasks that need to train custom dataset. There are few scenarios in which open-source datasets can be used as online models, so we need to carry out a series of operations on our custom datasets to ensure that the models can be put into production and serve users.

```{SeeAlso}
The video of this document has been posted on Bilibili: [A nanny level tutorials for custom datasets from annotationt to deployment](https://www.bilibili.com/video/BV1RG4y137i5)
```

```{Note}
All instructions in this document are done on Linux and are fully available on Windows, only slightly different in commands and operations.
```

Default that you have completed the installation of MMYOLO, if not installed, please refer to the document \[start your first step\] (https://mmyolo.readthedocs.io/zh_CN/latest/get_started.html#id1) for installation.

In this tutorial, we will introduce the whole process from annotating custom dataset to final training, testing and deployment. The overview steps are as below:

01. Prepare dataset: `tools/misc/download_dataset.py`
02. Use the software of [labelme](https://github.com/wkentaro/labelme) to annotate: `demo/image_demo.py` + labelme
03. Reorganize the dataset into COCO format: `tools/dataset_converters/labelme2coco.py`
04. Split dataset:`tools/misc/coco_split.py`
05. Creat a config file based on dataset
06. Dataset visualization analysis: `tools/analysis_tools/dataset_analysis.py`
07. Optimize Anchor size: `tools/analysis_tools/optimize_anchors.py`
08. Visualization the data processing part of config: `tools/analysis_tools/browse_dataset.py`
09. Train: `tools/train.py`
10. Inference: `demo/image_demo.py`
11. Deployment

```{Note}
After obtaining the model weight and the mAP of validation set, users need to deep analyse the  bad cases of incorrect predictions in order to optimize model. MMYOLO will add this function in the future. Expect.
```

Each step is described in detail below.

## 1. Prepare the customized dataset

- If you don't have your own dataset, or want to use a small dataset to run the whole process, you can use the 144 images `cat` dataset provided with this tutorial (the raw picture of this dataset is supplied by @RangeKing, cleaned by @PeterH0323). This `cat` dataset will be used as an example for the rest tutorial.

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205423220-c4b8f2fd-22ba-4937-8e47-1b3f6a8facd8.png" alt="cat dataset"/>
</div>

The download is also very simple, requiring only one command (dataset compression package size `217 MB`):

```shell
python tools/misc/download_dataset.py --dataset-name cat --save-dir ./data/cat --unzip --delete
```

This dataset is automatically downloaded to the `./data/cat` dir with the following directory structure:

```shell
.
└── ./data/cat
    ├── images # image files
    │    ├── image1.jpg
    │    ├── image2.png
    │    └── ...
    ├── labels # labelme files
    │    ├── image1.json
    │    ├── image2.json
    │    └── ...
    ├── annotations # annotated files of COCO
    │    ├── annotations_all.json # all labels of COCO
    │    ├── trainval.json # 80% labels of the dataset
    │    └── test.json # 20% labels of the dataset
    └── class_with_id.txt # id + class_name file
```

**Tips**：This dataset can be trained directly, and you can remove everything outside the `images` dir if you want to go through the whole process.

- If you already have a dataset, you can compose it into the following structure:

```shell
.
└── $DATA_ROOT
    └── images
         ├── image1.jpg
         ├── image2.png
         └── ...
```

## 2. Use the software of labelme to annotate \[to be update\]

In general, there are two annotation methods:

- Software or algorithmic assistance + manual correction
- Only manual annotation

## 2.1 Software or algorithmic assistance + manual correction

The principle is using the existing model to inference, and save the result as label file.

**Tips**：If the existing model doesn't have the categories defined in your dataset, such as COCO pre-trained model, you can manually annotate 100 images to train an initial model, and then software assistance.

Manually operating the software and loading the generated label files, you only need to check whether each image is correctly labeled and whether there are missing objects.

\[assistance + manual correction】you can save a lot of time in order to reduce costs and speed up by this way.

The process is described below:

### 2.1.1 Software or algorithmic assistance

MMYOLO provide model inference script `demo/image_demo.py`. Setting `--to-labelme` to generate labelme format label file:

```shell
python demo/image_demo.py img \
                          config \
                          checkpoint
                          [--out-dir OUT_DIR] \
                          [--device DEVICE] \
                          [--show] \
                          [--deploy] \
                          [--score-thr SCORE_THR] \
                          [--class-name CLASS_NAME]
                          [--to-labelme]
```

These include:

- `img`： image path, supported by dir, file, URL;
- `config`：config file path of model;
- `checkpoint`：weight file path of model;
- `--out-dir`：inference results saved in this dir, default as `./output`, if set this `--show` parameter, the detection results are not saved;
- `--device`：cumputing resources, including `CUDA`, `CPU` etc., default as `cuda:0`;
- `--show`：display the detection results, default as `False`；
- `--deploy`：whether to switch to deploy mode;
- `--score-thr`：confidence threshold, default as `0.3`;
- `--to-labelme`：whether to export label files in `labelme` format, shouldn't exist with the `--show` at the same time.

For example:

Here, we'll use YOLOv5-s as an example to help us label the 'cat' dataset we just downloaded. First, download the weights for YOLOv5-s:

```shell
mkdir work_dirs
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth -P ./work_dirs
```

Since the COCO 80 dataset already includes the `cat` class, we can directly load the COCO pre-trained model for assistant annotation.

```shell
python demo/image_demo.py ./data/cat/images \
                          ./configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                          ./work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
                          --out-dir ./data/cat/labels \
                          --class-name cat \
                          --to-labelme
```

**Tips**：

- If your dataset needs to label with multiclass, you can use this `--class-name class1 class2` format;
- Removing the `--class-name` flag to output all classes.

the generated label files saved in `--out-dir`:

```shell
.
└── $OUT_DIR
    ├── image1.json
    ├── image1.json
    └── ...
```

### 2.1.2 Manual annotation

In this tutorial, we use [labelme](https://github.com/wkentaro/labelme) to annotate

- Install labelme

```shell
pip install labelme
```

- Start labelme

```shell
labelme ${image dir path (same as the previous step)} \
        --output ${the dir path of label file（same as --out-dir）} \
        --autosave \
        --nodata
```

These include:

- `--output`：saved path of labelme file. If there already exists label file of some images, it will be loaded;
- `--autosave`：auto-save label file, and some tedioys steps will be omitted.
- `--nodata`：doesn't store the base64 encoding of each image, so setting this flag will greatly reduce the size of the label file.

For example：

```shell
labelme ./data/cat/images --output ./data/cat/labels --autosave --nodata
```

Type in command and labelme will start, and then check label. If labelme fails to start, type `export QT_DEBUG_PLUGINS=1` in command to see which libraries are missing and install it.

**Note: Make sure to use `rectangle` with the shortcut `Ctrl + R` (see below).**

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/204076212-86dab4fa-13dd-42cd-93d8-46b04b864449.png" alt="rectangle"/>
</div>

## 2.2 Only manual annotation

The procedure is the same as 【2.1.2 Manual annotation】, except that this is a direct labeling, there is no pre-generated label.
