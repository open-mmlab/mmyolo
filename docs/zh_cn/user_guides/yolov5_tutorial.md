# YOLOv5 从入门到部署全流程

## 环境安装

温馨提醒：由于本仓库采用的是 OpenMMLab 2.0，请最好新建一个 conda 虚拟环境，防止和 OpenMMLab 1.0 已经安装的仓库冲突。

```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch
# conda install pytorch torchvision cpuonly -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc1"
mim install "mmdet>=3.0.0rc0"

git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
pip install -v -e .
```

详细环境配置操作请查看 [get_started](../get_started.md)

## 数据集准备

本文选取不到 40MB 大小的 balloon 气球数据集作为 MMYOLO 的学习数据集。

```shell
python tools/misc/download_dataset.py  --dataset-name balloon --save-dir data --unzip
python tools/dataset_converters/balloon2coco.py
```

执行以上命令，下载数据集并转化格式后，balloon 数据集在 data 文件夹中准备好了，train.json 和 val.json 便是 coco 格式的标注文件了。

![](https://cdn.vansin.top/img/20220912105312.png)

## config文件准备

在 configs/yolov5 文件夹下新建 yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py 配置文件，并把以下内容复制配置文件中。

```python
_base_ = './yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = 'data/balloon/'

train_batch_size_per_gpu = 4
train_num_workers = 2

metainfo = {
    'CLASSES': ('balloon', ),
    'PALETTE': [
        (220, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='val/'),
        ann_file='val.json'))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val.json')

test_evaluator = val_evaluator

model = dict(bbox_head=dict(head_module=dict(num_classes=1)))

default_hooks = dict(logger=dict(interval=1))

```

以上配置从 `./yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py` 中继承，并根据 balloon 数据的特点更新了 `data_root`、`metainfo`、`train_dataloader`、`val_dataloader`、`num_classes` 等配置。
我们将 logger 的 interval 设置为 1 的原因是，每进行 interval 次 iteration 会输出一次 loss 相关的日志，而我们选取气球数据集比较小，interval 太大我们将看不到 loss 相关日志的输出。

## 训练

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py
```

运行以上训练命令，`work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon` 文件夹会被自动生成，权重文件以及此次的训练配置文件将会保存在此文件夹中。

![](https://cdn.vansin.top/img/20220913213846.png)

### 中断后恢复训练

如果训练中途停止，在训练命令最后加上 --resume ,程序会自动从 work_dirs 中加载最新的权重文件恢复训练。

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py --resume
```

### 加载预训练权重微调

经过测试，相比不加载预训练模型，加载 YOLOv5 官方的预训练模型在气球数据集上训练和验证 coco/bbox_mAP 能涨 30 多个百分点。

1. 下载 COCO 数据集预训练权重

```shell
cd mmyolo
wget xxx -O yolov5-6.1.zip
```

2. 加载预训练模型进行训练

```shell
cd mmyolo
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py --cfg-options load_from='yolov5-6.1/yolov5s.pth'
```

3. 冻结backbone进行训练

通过config文件或者命令行中设置 model.backbone.frozen_stages=4 冻结 backbone 的 4 个 stages。

```shell
# 命令行中设置 model.backbone.frozen_stages=4
cd mmyolo
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py --cfg-options load_from='yolov5-6.1/yolov5s.pth' model.backbone.frozen_stages=4
```

### 可视化训练参数

本教程以 wandb 展示 loss 等数据的可视化, wandb 官网注册并在 https://wandb.ai/settings 获取到 wandb 的 API Keys

![](https://cdn.vansin.top/img/20220913212628.png)

```shell
pip install wandb
# 运行了 wandb login 后输入上文中获取到的 API Keys ，便登录成功。
wandb login
```

在 `configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py` 添加 wandb 配置

```python
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
```

重新运行训练命令便可以在命令行中提示的网页链接中看到 loss、学习率和 coco/bbox_mAP 等数据可视化了。

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py
```

![](https://cdn.vansin.top/img/20220913213221.png)


## 部署

模型部署采用mmdeploy进行，提供了2种后端部署方式：onnxruntime后端的部署方式和TensorRT后端的部署形式，实际应用过程中选择一种即可。

### mmdeploy环境准备

mmdeploy环境准备请参见mmdeploy的 [入门文档](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/get_started.md)

### onnxruntime部署（onnx格式）
- 模型转换

在mmdeploy路径下执行下面的命令，如果程序运行成功，会生成mmyolo文件，里面包含生成的onnx中间表达：end2end.onnx。

```shell
cd /path/to/mmdeploy #(update to your own path)
python tools/deploy.py \
    configs/mmyolo/detection/detection_onnxruntime_static-640x640.py \  # 导出的onnx配置参数
    /path/to/mmyolo/configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py  \  # #(update to your own path)
    /path/to/mmyolo/checkpoints \  # #(update to your own path)
    /path/to/mmyolo/demo/dog.jpg \  # #(update to your own path)
    --work-dir mmyolo   # 导出的onnx文件目录
```

- 用onnxruntime进行推理部署

将end2end.onnx的中间表达，使用ORTWrapper加载，完成推理过程，完整代码如下：

```python
import torch
import cv2
import random
import numpy as np
from mmdeploy.backend.onnxruntime import ORTWrapper
from PIL import Image
from pathlib import Path

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im_copy = im.copy()
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, im_copy, r, (dw, dh)

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)}

cv2.setNumThreads(0)
device = torch.device('cuda:0')
engine_file = 'mmyolo/end2end.onnx'
model = ORTWrapper(engine_file)
image_path = Path('data/coco/train2017')

for i in image_path.glob('*.jpg'):

    image = cv2.imread(str(i))
    image, image_orin, ratio, dwdh = letterbox(image, auto=False)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_Copy = image.copy()

    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = torch.from_numpy(image.astype(np.float32))
    im/=255


    inputs = dict(input=im.to(device))
    outputs = model(inputs)

    # 后处理放这里
    outputs['dets'][:, :, :4] -= torch.tensor([dwdh * 2], device=device, dtype=torch.float32)

    for (x0,y0,x1,y1,conf),cls in zip(outputs['dets'][0],outputs['labels'][0]):
        name = names[int(cls)]
        color = colors[name]
        cv2.rectangle(image_orin, [int(x0),int(y0)], [int(x1),int(y1)], color, 2)
        cv2.putText(image_orin, name, (int(x0), int(y0) - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, [225, 255, 255], thickness=2)

    cv2.imshow('win',image_orin)
    cv2.waitKey(0)
```

### TensorRT部署（engine格式）

- 模型转换
在mmdeploy路径下执行下面的命令，如果程序运行成功，会生成mmyolo文件，里面包含end2end.onnx和end2end.engine，其中end2end.engine是用于trt推理。

```shell
cd /path/to/mmdeploy #(update to your own path)
python tools/deploy.py \
    configs/mmyolo/detection/detection_tensorrt_static-640x640.py \  # 导出的onnx配置参数
    /path/to/mmyolo/configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py  \  # #(update to your own path)
    /path/to/mmyolo/checkpoints \  # #(update to your own path)
    /path/to/mmyolo/demo/dog.jpg \  # #(update to your own path)
    --work-dir mmyolo   # 导出的onnx文件目录
```

- 用TensorRT进行推理部署

使用TRTWrapper加载end2end.engine，完成推理过程，完整代码如下：

```python
import torch
import cv2
import random
import numpy as np
from mmdeploy.backend.tensorrt import TRTWrapper
from PIL import Image
from pathlib import Path

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im_copy = im.copy()
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, im_copy, r, (dw, dh)

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)}

cv2.setNumThreads(0)
device = torch.device('cuda:0')
engine_file = 'mmyolo/end2end.engine'
model = TRTWrapper(engine_file)
image_path = Path('data/coco/train2017')

for i in image_path.glob('*.jpg'):

    image = cv2.imread(str(i))
    image, image_orin, ratio, dwdh = letterbox(image, auto=False)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_Copy = image.copy()

    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = torch.from_numpy(image.astype(np.float32))
    im/=255


    inputs = dict(input=im.to(device))
    outputs = model(inputs)

    # 后处理放这里
    outputs['dets'][:, :, :4] -= torch.tensor([dwdh * 2], device=device, dtype=torch.float32)

    for (x0,y0,x1,y1,conf),cls in zip(outputs['dets'][0],outputs['labels'][0]):
        name = names[int(cls)]
        color = colors[name]
        cv2.rectangle(image_orin, [int(x0),int(y0)], [int(x1),int(y1)], color, 2)
        cv2.putText(image_orin, name, (int(x0), int(y0) - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, [225, 255, 255], thickness=2)

    cv2.imshow('win',image_orin)
    cv2.waitKey(0)
```
