# Convert Model

The six scripts under the `tools/model_converters` directory can help users convert the keys in the official pre-trained model of YOLO to the format of MMYOLO, and use MMYOLO to fine-tune the model.

## YOLOv5

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

## YOLOX

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
