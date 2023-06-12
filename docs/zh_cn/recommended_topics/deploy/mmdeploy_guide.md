# MMDeploy 部署

## MMDeploy 介绍

MMDeploy 是 [OpenMMLab](https://openmmlab.com/) 模型部署工具箱，**为各算法库提供统一的部署体验**。基于 MMDeploy，开发者可以轻松从训练 repo 生成指定硬件所需 SDK，省去大量适配时间。

更多介绍和使用指南见 https://mmdeploy.readthedocs.io/zh_CN/latest/get_started.html

## 算法支持列表

目前支持的 model-backend 组合：

| Model  | Task            | OnnxRuntime | TensorRT |                              Model config                               |
| :----- | :-------------- | :---------: | :------: | :---------------------------------------------------------------------: |
| YOLOv5 | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov5) |
| YOLOv6 | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov6) |
| YOLOX  | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolox)  |
| RTMDet | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet) |

ncnn 和其他后端的支持会在后续支持。

## 安装

按照[说明](https://mmdeploy.readthedocs.io/zh_CN/latest/get_started.html)安装 mmdeploy。

```{note}
如果安装的是 mmdeploy 预编译包，那么也请通过 ‘git clone https://github.com/open-mmlab/mmdeploy.git –depth=1’ 下载 mmdeploy 源码。因为它包含了部署时所需的 tools 文件夹。
```

## MMYOLO 中部署相关配置说明

所有部署配置文件在 [`configs/deploy`](../../../configs/deploy/) 目录下。

您可以部署静态输入或者动态输入的模型，因此您需要修改模型配置文件中与此相关的数据处理流程。

MMDeploy 将后处理整合到自定义的算子中，因此您可以修改 `codebase_config` 中的 `post_processing` 参数来调整后处理策略，参数描述如下：

```python
codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1),
    module=['mmyolo.deploy'])
```

- `score_threshold`：在 `nms` 之前筛选候选框的类别分数阈值。
- `confidence_threshold`：在 `nms` 之前筛选候选框的置信度分数阈值。
- `iou_threshold`：在 `nms` 中去除重复框的 `iou` 阈值。
- `max_output_boxes_per_class`：每个类别最大的输出框数量。
- `pre_top_k`：在 `nms` 之前对候选框分数排序然后固定候选框的个数。
- `keep_top_k`：`nms` 算法最终输出的候选框个数。
- `background_label_id`：MMYOLO 算法中没有背景类别信息，置为 `-1` 即可。

### 静态输入配置

#### (1) 模型配置文件介绍

以 MMYOLO 中的 `YOLOv5` 模型配置为例，下面是对部署时使用的模型配置文件参数说明介绍。

```python
_base_ = '../../yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=False,
        use_mini_pad=False,
    ),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

test_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, batch_shapes_cfg=None))
```

`_base_ = '../../yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'` 继承了训练时构建模型的配置。

`test_pipeline` 为部署时对输入图像进行处理的流程，`LetterResize` 控制了输入图像的尺寸，同时限制了导出模型所能接受的输入尺寸。

`test_dataloader` 为部署时构建数据加载器配置，`batch_shapes_cfg` 控制了是否启用 `batch_shapes` 策略，详细内容可以参考 [yolov5 配置文件说明](../../tutorials/config.md) 。

#### (2) 部署配置文件介绍

以 `MMYOLO` 中的 `YOLOv5` 部署配置为例，下面是对配置文件参数说明介绍。

`ONNXRuntime` 部署 `YOLOv5` 可以使用 [`detection_onnxruntime_static.py`](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy/detection_onnxruntime_static.py) 配置。

```python
_base_ = ['./base_static.py']
codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1),
    module=['mmyolo.deploy'])
backend_config = dict(type='onnxruntime')
```

`backend_config` 中指定了部署后端 `type='onnxruntime'`，其他信息可参考第三小节。

`TensorRT` 部署 `YOLOv5` 可以使用 [`detection_tensorrt_static-640x640.py`](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy/detection_tensorrt_static-640x640.py) 配置。

```python
_base_ = ['./base_static.py']
onnx_config = dict(input_shape=(640, 640))
backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=False, max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 640, 640],
                    opt_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 640, 640])))
    ])
use_efficientnms = False
```

`backend_config` 中指定了后端 `type=‘tensorrt’`。

与 `ONNXRuntime` 部署配置不同的是，`TensorRT`  需要指定输入图片尺寸和构建引擎文件需要的参数，包括：

- `onnx_config` 中指定 `input_shape=(640, 640)`
- `backend_config['common_config']` 中包括 `fp16_mode=False` 和 `max_workspace_size=1 << 30`, `fp16_mode` 表示是否以 `fp16` 的参数格式构建引擎，`max_workspace_size` 表示当前 `gpu` 设备最大显存, 单位为 `GB`。`fp16` 的详细配置可以参考 [`detection_tensorrt-fp16_static-640x640.py`](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy/detection_tensorrt-fp16_static-640x640.py)
- `backend_config['model_inputs']['input_shapes']['input']` 中 `min_shape` /`opt_shape`/`max_shape` 对应的值在静态输入下应该保持相同，即默认均为 `[1, 3, 640, 640]`。

`use_efficientnms` 是 `MMYOLO` 系列新引入的配置，表示在导出 `onnx` 时是否启用`Efficient NMS Plugin`来替换 `MMDeploy` 中的 `TRTBatchedNMS plugin` 。

可以参考 `TensorRT` 官方实现的 [Efficient NMS Plugin](https://github.com/NVIDIA/TensorRT/blob/main/plugin/efficientNMSPlugin/README.md) 获取更多详细信息。

**注意**，这个功能仅仅在 TensorRT >= 8.0 版本才能使用，无需编译开箱即用。

### 动态输入配置

#### (1) 模型配置文件介绍

当您部署动态输入模型时，您无需修改任何模型配置文件，仅需要修改部署配置文件即可。

#### (2) 部署配置文件介绍

`ONNXRuntime` 部署 `YOLOv5` 可以使用  [`detection_onnxruntime_dynamic.py`](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy/detection_onnxruntime_dynamic.py)  配置。

```python
_base_ = ['./base_dynamic.py']
codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1),
    module=['mmyolo.deploy'])
backend_config = dict(type='onnxruntime')
```

`backend_config` 中指定了后端 `type='onnxruntime'`，其他配置与上一节在 ONNXRuntime 部署静态输入模型相同。

`TensorRT` 部署 `YOLOv5` 可以使用  [`detection_tensorrt_dynamic-192x192-960x960.py`](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy/detection_tensorrt_dynamic-192x192-960x960.py) 配置。

```python
_base_ = ['./base_dynamic.py']
backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=False, max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 192, 192],
                    opt_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 960, 960])))
    ])
use_efficientnms = False
```

`backend_config` 中指定了后端 `type='tensorrt'`，由于 `TensorRT` 动态输入与静态输入有所不同，您可以了解更多动态输入相关信息通过访问 [TensorRT dynamic input official introduction](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/developer-guide/index.html#work_dynamic_shapes)。

`TensorRT` 部署需要配置 `min_shape`, `opt_shape`, `max_shape` ，`TensorRT` 限制输入图片的尺寸在 `min_shape` 和 ` max_shape` 之间。

`min_shape` 为输入图片的最小尺寸，`opt_shape` 为输入图片常见尺寸, 在这个尺寸下推理性能最好，`max_shape` 为输入图片的最大尺寸。

`use_efficientnms` 配置与上节 `TensorRT` 静态输入配置相同。

### INT8 量化配置

!!! 部署 TensorRT INT8 模型教程即将发布 !!!

## 模型转换

### 使用方法

#### 从源码安装的 MMDeploy

设置 `MMDeploy` 根目录为环境变量 `MMDEPLOY_DIR` ，例如 `export MMDEPLOY_DIR=/the/root/path/of/MMDeploy`

```shell
python3 ${MMDEPLOY_DIR}/tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    ${INPUT_IMG} \
    --test-img ${TEST_IMG} \
    --work-dir ${WORK_DIR} \
    --calib-dataset-cfg ${CALIB_DATA_CFG} \
    --device ${DEVICE} \
    --log-level INFO \
    --show \
    --dump-info
```

### 参数描述

- `deploy_cfg` : mmdeploy 针对此模型的部署配置，包含推理框架类型、是否量化、输入 shape 是否动态等。配置文件之间可能有引用关系，`configs/deploy/detection_onnxruntime_static.py` 是一个示例。
- `model_cfg` : MMYOLO 算法库的模型配置，例如 `configs/deploy/model/yolov5_s-deploy.py`, 与 mmdeploy 的路径无关。
- `checkpoint` : torch 模型路径。可以 http/https 开头，详见 `mmengine.fileio` 的实现。
- `img` : 模型转换时，用做测试的图像文件路径。
- `--test-img` : 用于测试模型的图像文件路径。默认设置成`None`。
- `--work-dir` : 工作目录，用来保存日志和模型文件。
- `--calib-dataset-cfg` : 此参数只有int8模式下生效，用于校准数据集配置文件。若在int8模式下未传入参数，则会自动使用模型配置文件中的'val'数据集进行校准。
- `--device` : 用于模型转换的设备。 默认是`cpu`，对于 trt 可使用 `cuda:0` 这种形式。
- `--log-level` : 设置日记的等级，选项包括`'CRITICAL'， 'FATAL'， 'ERROR'， 'WARN'， 'WARNING'， 'INFO'， 'DEBUG'， 'NOTSET'`。 默认是`INFO`。
- `--show` : 是否显示检测的结果。
- `--dump-info` : 是否输出 SDK 信息。

#### 通过 pip install 安装的 MMDeploy

假设当前的工作目录为 mmyolo 的根目录, 那么以 [YoloV5](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py) 模型为例，你可以从[此处](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth)下载对应的 checkpoint，并使用以下代码将之转换为 onnx 模型：

```python
from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

img = 'demo/demo.jpg'
work_dir = 'mmdeploy_models/mmyolo/onnx'
save_file = 'end2end.onnx'
deploy_cfg = 'configs/deploy/detection_onnxruntime_dynamic.py'
model_cfg = 'configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'
model_checkpoint = 'checkpoints/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'
device = 'cpu'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
           model_checkpoint, device)

# 2. extract pipeline info for inference by MMDeploy SDK
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint,
           device=device)
```

## 模型规范

在使用转换后的模型进行推理之前，有必要了解转换结果的结构。 它存放在 `--work-dir` 指定的路路径下。

上例中的`mmdeploy_models/mmyolo/onnx`，结构如下：

```
mmdeploy_models/mmyolo/onnx
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

重要的是：

- **end2end.onnx**: 推理引擎文件。可用 ONNX Runtime 推理
- ***xxx*.json**:  mmdeploy SDK 推理所需的 meta 信息

整个文件夹被定义为**mmdeploy SDK model**。换言之，**mmdeploy SDK model**既包括推理引擎，也包括推理 meta 信息。

## 模型推理

### 后端模型推理

以上述模型转换后的 `end2end.onnx` 为例，你可以使用如下代码进行推理：

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/deploy/detection_onnxruntime_dynamic.py'
model_cfg = 'configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'
device = 'cpu'
backend_model = ['mmdeploy_models/mmyolo/onnx/end2end.onnx']
image = 'demo/demo.jpg'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='work_dir/output_detection.png')
```

运行上述代码后，你可以在 `work_dir` 中看到推理的结果图片 `output_detection.png`。

### SDK模型推理

你也可以参考如下代码，对 SDK model 进行推理：

```python
from mmdeploy_runtime import Detector
import cv2

img = cv2.imread('demo/demo.jpg')

# create a detector
detector = Detector(model_path='mmdeploy_models/mmyolo/onnx',
                    device_name='cpu', device_id=0)
# perform inference
bboxes, labels, masks = detector(img)

# visualize inference result
indices = [i for i in range(len(bboxes))]
for index, bbox, label_id in zip(indices, bboxes, labels):
    [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
    if score < 0.3:
        continue

    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

cv2.imwrite('work_dir/output_detection.png', img)
```

除了python API，mmdeploy SDK 还提供了诸如 C、C++、C#、Java等多语言接口。
你可以参考[样例](https://github.com/open-mmlab/mmdeploy/tree/main/demo)学习其他语言接口的使用方法。

## 模型评测

当您将 PyTorch 模型转换为后端支持的模型后，您可能需要验证模型的精度，使用 `${MMDEPLOY_DIR}/tools/test.py`

```shell
python3 ${MMDEPLOY_DIR}/tools/test.py \
    ${DEPLOY_CFG} \
    ${MODEL_CFG} \
    --model ${BACKEND_MODEL_FILES} \
    --device ${DEVICE} \
    --work-dir ${WORK_DIR} \
    [--cfg-options ${CFG_OPTIONS}] \
    [--show] \
    [--show-dir ${OUTPUT_IMAGE_DIR}] \
    [--interval ${INTERVAL}] \
    [--wait-time ${WAIT_TIME}] \
    [--log2file work_dirs/output.txt]
    [--speed-test] \
    [--warmup ${WARM_UP}] \
    [--log-interval ${LOG_INTERVERL}] \
    [--batch-size ${BATCH_SIZE}] \
    [--uri ${URI}]
```

### 参数描述

- `deploy_cfg`: 部署配置文件。
- `model_cfg`: MMYOLO 模型配置文件。
- `--model`: 导出的后端模型。 例如, 如果我们导出了 TensorRT 模型，我们需要传入后缀为 ".engine" 文件路径。
- `--device`: 运行模型的设备。请注意，某些后端会限制设备。例如，TensorRT 必须在 cuda 上运行。
- `--work-dir`: 模型转换、报告生成的路径。
- `--cfg-options`: 传入额外的配置，将会覆盖当前部署配置。
- `--show`: 是否在屏幕上显示评估结果。
- `--show-dir`: 保存评估结果的目录。(只有给出这个参数才会保存结果)。
- `--interval`: 屏幕上显示评估结果的间隔。
- `--wait-time`: 每个窗口的显示时间
- `--log2file`: 将评估结果（和速度）记录到文件中。
- `--speed-test`: 是否开启速度测试。
- `--warmup`: 在计算推理时间之前进行预热，需要先开启 `speed-test`。
- `--log-interval`: 每个日志之间的间隔，需要先设置 `speed-test`。
- `--batch-size`: 推理的批量大小，它将覆盖数据配置中的 `samples_per_gpu`。默认为 `1`。请注意，并非所有模型都支持 `batch_size > 1`。
- `--uri`: 在边缘设备上推理时的 ipv4 或 ipv6 端口号。

注意：`${MMDEPLOY_DIR}/tools/test.py` 中的其他参数用于速度测试。他们不影响评估。
