# Useful tools

We provide lots of useful tools under the `tools/` directory. In addition, you can also quickly run other open source libraries of OpenMMLab through MIM.

Take MMDetection as an example. If you want to use [print_config.py](https://github.com/open-mmlab/mmdetection/blob/3.x/tools/misc/print_config.py), you can directly use the following commands without copying the source code to the MMYOLO library.

```shell
mim run mmdet print_config [CONFIG]
```

**Note**: The MMDetection library must be installed through the MIM before the above command can succeed.

## Visualization

### Visualize COCO labels

`tools/analysis_tools/browse_coco_json.py` is a script that can visualization to display the COCO label in the picture.

```shell
python tools/analysis_tools/browse_coco_json.py ${DATA_ROOT} \
                                                [--ann_file ${ANN_FILE}] \
                                                [--img_dir ${IMG_DIR}] \
                                                [--wait-time ${WAIT_TIME}] \
                                                [--disp-all] [--category-names CATEGORY_NAMES [CATEGORY_NAMES ...]] \
                                                [--shuffle]
```

E.g:

1. Visualize all categories of `COCO` and display all types of annotations such as `bbox` and `mask`:

```shell
python tools/analysis_tools/browse_coco_json.py './data/coco/' \
                                                --ann_file 'annotations/instances_train2017.json' \
                                                --img_dir 'train2017' \
                                                --disp-all
```

2. Visualize all categories of `COCO`, and display only the `bbox` type labels, and shuffle the image to show:

```shell
python tools/analysis_tools/browse_coco_json.py './data/coco/' \
                                                --ann_file 'annotations/instances_train2017.json' \
                                                --img_dir 'train2017' \
                                                --shuffle
```

3. Only visualize the `bicycle` and `person` categories of `COCO` and only the `bbox` type labels are displayed:

```shell
python tools/analysis_tools/browse_coco_json.py './data/coco/' \
                                                --ann_file 'annotations/instances_train2017.json' \
                                                --img_dir 'train2017' \
                                                --category-names 'bicycle' 'person'
```

4. Visualize all categories of `COCO`, and display all types of label such as `bbox`, `mask`, and shuffle the image to show:

```shell
python tools/analysis_tools/browse_coco_json.py './data/coco/' \
                                                --ann_file 'annotations/instances_train2017.json' \
                                                --img_dir 'train2017' \
                                                --disp-all \
                                                --shuffle
```

### Visualize Datasets

`tools/analysis_tools/browse_dataset.py` helps the user to browse a detection dataset (both images and bounding box annotations) visually, or save the image to a designated directory.

```shell
python tools/analysis_tools/browse_dataset.py ${CONFIG} \
                                              [-h] \
                                              [--output-dir ${OUTPUT_DIR}] \
                                              [--not-show] \
                                              [--show-interval ${SHOW_INTERVAL}]
```

E,g：

1. Use `config` file `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py` to visualize the picture. The picture will pop up directly and be saved to the directory `work dir/browse_ dataset` at the same time:

```shell
python tools/analysis_tools/browse_dataset.py 'configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py' \
                                               --output-dir 'work-dir/browse_dataset'
```

2. Use `config` file `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py` to visualize the picture. The picture will pop up and display directly. Each picture lasts for `10` seconds. At the same time, it will be saved to the directory `work dir/browse_ dataset`:

```shell
python tools/analysis_tools/browse_dataset.py 'configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py' \
                                               --output-dir 'work-dir/browse_dataset' \
                                               --show-interval 10
```

3. Use `config` file `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py` to visualize the picture. The picture will pop up and display directly. Each picture lasts for `10` seconds and the picture will not be saved:

```shell
python tools/analysis_tools/browse_dataset.py 'configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py' \
                                               --show-interval 10
```

4. Use `config` file `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py` to visualize the picture. The picture will not pop up directly, but only saved to the directory `work dir/browse_ dataset`:

```shell
python tools/analysis_tools/browse_dataset.py 'configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py' \
                                               --output-dir 'work-dir/browse_dataset' \
                                               --not-show
```

## Convert Dataset

The folder `tools/data_converters` currently contains `ballon2coco.py` and `yolo2coco.py` two dataset conversion tools.

- `ballon2coco.py` converts the `balloon` dataset (this small dataset is for starters only) to COCO format.

For a detailed description of this script, please see the `Dataset Preparation` section in [YOLOv5 from entry to deployment](./yolov5_tutorial.md).

```shell
python tools/dataset_converters/balloon2coco.py
```

- `yolo2coco.py` converts a dataset from `yolo-style` **.txt** format to COCO format, please use it as follows:

```shell
python tools/dataset_converters/yolo2coco.py --image-dir /path/to/the/root/dir/of/your_dataset [--split]
```

Instructions:

1. `image-dir` is the root directory of the yolo-style dataset, which should contain `images`, `labels`, and `class.txt`. `class.txt` is the class declaration corresponding to the current dataset. One class a line. The structure of the root directory should be formatted as this example shows:

```bash
.
└── $ROOT_PATH
         ├── class.txt
         ├── labels
             ├── a.txt
             ├── b.txt
         ├── images
             ├── a.jpg
             ├── b.png
         └── ...
```

2. The script will automatically check if `train.txt`, `val.txt`, and `test.txt` are already existed under `image-dir`. If these files are located, the script will organize the dataset accordingly. Otherwise, the script will convert the dataset in one file.
3. By default, the script will create a folder called `coco_format` in the `image-dir` directory where stores the converted JSON file. If `train.txt`, `val.txt`, and `test.txt` are not found, the output file is `result.json`. Otherwise, three corresponding files called `train.json`, `val.json`, and `test.json` will be generated.

## Download Dataset

`tools/misc/download_dataset.py` supports downloading datasets such as `COCO`, `VOC`, `LVIS` and `Balloon`.

```shell
python tools/misc/download_dataset.py --dataset-name coco2017
python tools/misc/download_dataset.py --dataset-name voc2007
python tools/misc/download_dataset.py --dataset-name lvis
python tools/misc/download_dataset.py --dataset-name balloon [--save-dir ${SAVE_DIR}] [--unzip]
```

## Convert Model

The three scripts under the `tools/` directory can help users convert the keys in the official pre-trained model of YOLO to the format of MMYOLO, and use MMYOLO to fine tune the model.

### YOLOv5

Take conversion `yolov5s.pt` as an example:

1. Clone the official YOLOv5 code to the local (currently the maximum supported version is `v6.1`):

```shell
git clone -b v6.1 https://github.com/ultralytics/yolov5.git
cd yolov5
```

2. Download official weight file:

```shell
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
```

3. Copy file `tools/model_converters/yolov5_to_mmyolo.py` to the path of YOLOv5 official code clone:

```shell
cp ${MMDET_YOLO_PATH}/tools/model_converters/yolov5_to_mmyolo.py yolov5_to_mmyolo.py
```

4. Conversion

```shell
python yolov5_to_mmyolo.py --src ${WEIGHT_FILE_PATH} --dst mmyolov5.pt
```

The converted `mmyolov5.pt` can be used by MMYOLO. The official weight conversion of YOLOv6 is also used in the same way.

### YOLOX

The conversion of YOLOX model **does not need** to download the official YOLOX code, just download the weight.

Take conversion `yolox_s.pth` as an example:

1. Download official weight file:

```shell
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

2. Conversion

```shell
python tools/model_converters/yolox_to_mmyolo.py --src yolox_s.pth --dst mmyolox.pt
```

The converted `mmyolox.pt` can be used by MMYOLO.
