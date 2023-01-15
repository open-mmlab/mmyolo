# Training on a single channel image dataset

## Introduction

This project was developed for training on a single channel image dataset.

Now, the script only support `YOLOv5` .

## Usage

### Data set preparation

If you are using a custom grayscale image dataset, you can skip this step.

```shell
python tools/misc/download_dataset.py --dataset-name balloon --save-dir projects/single_channel/data --unzip
python projects/single_channel/balloon2coco_single_channel.py
```

Run the following command to replace the dataset with a single channel dataset.

```shell
python projects/single_channel/single_channel.py --path projects/single_channel/data/balloon
# --path  *Original dataset path
```

### Training

In the `yolov5_s-v61_syncbn_fast_1xb4-300e_balloon_single_channel.py`, `_base_` and `data_root` need to be modified to the corresponding dataset paths. There should be `train`, `train.json`, `val`, `val.json` under the dataset file.

```shell
python tools/train.py projects/single_channel/configs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon_single_channel.py
```

### Model Testing

```shell
python tools/test.py projects/single_channel/configs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon_single_channel.py \
                     work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon_single_channel/epoch_300.pth \
                     --show-dir show_results
```

<img src="https://raw.githubusercontent.com/landhill/mmyolo/main/resources/single_channel_test.jpg"/>

The left picture shows the physical labeling, and the right picture shows the target detection results.
