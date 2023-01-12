# YOLOv8

<!-- [ALGORITHM] -->

## Abstract

Ultralytics YOLOv8, developed by Ultralytics, is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection, image segmentation and image classification tasks.

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/211810400-a12a01f5-39fc-414f-85a7-8dead9f0c232.jpg"/>
YOLOv8-P5 model structure
</div>

## Results and models

### COCO

| Backbone | Arch | size | AMP | Mem (GB) | box AP |                                                  Config                                                   |        Download        |
| :------: | :--: | :--: | :-: | :------: | :----: | :-------------------------------------------------------------------------------------------------------: | :--------------------: |
| YOLOv8-n |  P5  | 640  | Yes |    x     |  37.3  | [config](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov8/yolov8_n_syncbn_8xb16-500e_coco.py) | [model](x) \| [log](x) |
| YOLOv8-s |  P5  | 640  | Yes |    x     |  44.9  | [config](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov8/yolov8_s_syncbn_8xb16-500e_coco.py) | [model](x) \| [log](x) |
| YOLOv8-m |  P5  | 640  | Yes |    x     |  50.3  | [config](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov8/yolov8_m_syncbn_8xb16-500e_coco.py) | [model](x) \| [log](x) |
| YOLOv8-l |  P5  | 640  | Yes |    x     |  52.8  | [config](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov8/yolov8_l_syncbn_8xb16-500e_coco.py) | [model](x) \| [log](x) |
| YOLOv8-x |  P5  | 640  | Yes |    x     |  53.8  | [config](https://github.com/open-mmlab/mmyolo/tree/dev/configs/yolov8/yolov8_x_syncbn_8xb16-500e_coco.py) | [model](x) \| [log](x) |

**Note**: The above AP is the result of the test after using the official weight conversion. We provide the [yolov8_to_mmyolo](https://github.com/open-mmlab/mmyolo/tree/dev/tools/model_converters/yolov8_to_mmyolo.py) script for you to convert YOLOv8 weights to MMYOLO.

## Citation
