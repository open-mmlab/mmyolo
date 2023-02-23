# MMYOLO Model Assigner Visualization

<img src="https://user-images.githubusercontent.com/40284075/208255302-dbcf8cb0-b9d1-495f-8908-57dd2370dba8.png"/>

## Introduction

This project is developed for easily showing assigning results. The script allows users to analyze where and how many positive samples each gt is assigned in the image.

Now, the script supports `YOLOv5`, `YOLOv7`, `YOLOv8` and `RTMDet`.

## Usage

### Command

YOLOv5 assigner visualization command:

```shell
python projects/assigner_visualization/assigner_visualization.py projects/assigner_visualization/configs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_assignervisualization.py
```

Note: `YOLOv5` does not need to load the trained weights.

YOLOv7 assigner visualization command:

```shell
python projects/assigner_visualization/assigner_visualization.py projects/assigner_visualization/configs/yolov7_tiny_syncbn_fast_8xb16-300e_coco_assignervisualization.py -c ${checkpont}
```

YOLOv8 assigner visualization command:

```shell
python projects/assigner_visualization/assigner_visualization.py projects/assigner_visualization/configs/yolov8_s_syncbn_fast_8xb16-500e_coco_assignervisualization.py  -c ${checkpont}
```

RTMdet assigner visualization command:

```shell
python projects/assigner_visualization/assigner_visualization.py projects/assigner_visualization/configs/rtmdet_s_syncbn_fast_8xb32-300e_coco_assignervisualization.py -c ${checkpont}
```

${checkpont} is the checkpont file path. Dynamic label assignment is used in `YOLOv7`, `YOLOv8` and `RTMDet`, model weights will affect the positive sample allocation results, so it is recommended to load the trained model weights.

If you want to know details about label assignment, you can check the [RTMDet](https://mmyolo.readthedocs.io/zh_CN/latest/algorithm_descriptions/rtmdet_description.html#id5).
