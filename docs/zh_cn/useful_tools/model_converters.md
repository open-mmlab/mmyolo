# 模型转换

文件夹 `tools/model_converters/` 下的六个脚本能够帮助用户将对应YOLO官方的预训练模型中的键转换成 `MMYOLO` 格式，并使用 `MMYOLO` 对模型进行微调。

## YOLOv5

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

## YOLOX

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
