
## Introduction

This project was developed for training on a single channel image dataset.

Now, the script only support `YOLOv5` .

## Usage

### Data set preparation

If you are using a custom grayscale image dataset, you can skip this step.

```shell
python tools/misc/download_dataset.py --dataset-name balloon --save-dir projects/single_channel/data --unzip
python projects/single_channel/balloon2coco_single_channel.py
python projects/single_channel/single_channel.py
```

### Training

In the configuration file, `_base_` and `data_root` need to be modified to the corresponding dataset paths.
```shell
python tools/train.py projects/single_channel/configs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon_single_channel.py
```
### Model Testing
```shell
python tools/test.py projects/single_channel/configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py \
                     work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon/epoch_300.pth \
                     --show-dir show_results
```