# MMYOLO Model Assigner Visualization

<img src="https://user-images.githubusercontent.com/40284075/208255302-dbcf8cb0-b9d1-495f-8908-57dd2370dba8.png"/>

## Introduction

This project is developed for easily showing assigning results. The script allows users to analyze where and how many positive samples each gt is assigned in the image.

Now, the script only support `YOLOv5` .

## Usage

### Command

```shell
python projects/assigner_visualization/assigner_visualization.py projects/assigner_visualization/configs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_assignervisualization.py
```
