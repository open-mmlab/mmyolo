# Basic Deployment Guide

## Introduction of MMDeploy

MMDeploy is an open-source deep learning model deployment toolset. It is a part of the [OpenMMLab](https://openmmlab.com/) project, and provides **a unified experience of exporting different models** to various platforms and devices of the OpenMMLab series libraries. Using MMDeploy, developers can easily export the specific compiled SDK they need from the training result, which saves a lot of effort.

More detailed introduction and guides can be found [here](https://mmdeploy.readthedocs.io/en/latest/get_started.html)

## Supported Algorithms

Currently our deployment kit supports on the following models and backends:

| Model  | Task            | OnnxRuntime | TensorRT |                              Model config                               |
| :----- | :-------------- | :---------: | :------: | :---------------------------------------------------------------------: |
| YOLOv5 | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov5) |
| YOLOv6 | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov6) |
| YOLOX  | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolox)  |
| RTMDet | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet) |

Note: ncnn and other inference backends support are coming soon.

## Installation

Please install mmdeploy by following [this](https://mmdeploy.readthedocs.io/en/latest/get_started.html) guide.

```{note}
If you install mmdeploy prebuilt package, please also clone its repository by 'git clone https://github.com/open-mmlab/mmdeploy.git --depth=1' to get the 'tools' file for deployment.
```

## How to Write Config for MMYOLO

All config files related to the deployment are located at [`configs/deploy`](../../../configs/deploy/).

You only need to change the relative data processing part in the model config file to support either static or dynamic input for your model. Besides, MMDeploy integrates the post-processing parts as customized ops, you can modify the strategy in `post_processing` parameter in `codebase_config`.

Here is the detail description:

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

- `score_threshold`: set the score threshold to filter candidate bboxes before `nms`
- `confidence_threshold`: set the confidence threshold to filter candidate bboxes before `nms`
- `iou_threshold`: set the `iou` threshold for removing duplicates in `nms`
- `max_output_boxes_per_class`: set the maximum number of bboxes for each class
- `pre_top_k`: set the number of fixedcandidate bboxes before `nms`, sorted by scores
- `keep_top_k`: set the number of output candidate bboxs after `nms`
- `background_label_id`: set to `-1` as MMYOLO has no background class information

### Configuration for Static Inputs

#### 1. Model Config

Taking `YOLOv5` of MMYOLO as an example, here are the details:

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

`_base_ = '../../yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'` inherits the model config in the training stage.

`test_pipeline` adds the data processing piple for the deployment, `LetterResize` controls the size of the input images and the input for the converted model

`test_dataloader` adds the dataloader config for the deployment, `batch_shapes_cfg` decides whether to use the `batch_shapes` strategy. More details can be found at [yolov5 configs](../user_guides/config.md)

#### 2. Deployment Config

Here we still use the `YOLOv5` in MMYOLO as the example. We can use [`detection_onnxruntime_static.py`](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy/detection_onnxruntime_static.py) as the config to deploy `YOLOv5` to `ONNXRuntime` with static inputs.

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

`backend_config` indicates the deployment backend with `type='onnxruntime'`, other information can be referred from the third section.

To deploy the `YOLOv5` to `TensorRT`, please refer to the [`detection_tensorrt_static-640x640.py`](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy/detection_tensorrt_static-640x640.py) as follows.

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

`backend_config` indices the backend with `type='tensorrt'`.

Different from `ONNXRuntime` deployment configuration, `TensorRT` needs to specify the input image size and the parameters required to build the engine file, including:

- `onnx_config` specifies the input shape as `input_shape=(640, 640)`
- `fp16_mode=False` and `max_workspace_size=1 << 30` in `backend_config['common_config']` indicates whether to build the engine in the parameter format of `fp16`, and the maximum video memory for the current `gpu` device, respectively. The unit is in `GB`. For detailed configuration of `fp16`, please refer to the [`detection_tensorrt-fp16_static-640x640.py`](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy/detection_tensorrt-fp16_static-640x640.py)
- The `min_shape`/`opt_shape`/`max_shape` in `backend_config['model_inputs']['input_shapes']['input']` should remain the same under static input, the default is `[1, 3, 640, 640]`.

`use_efficientnms` is a new configuration introduced by the `MMYOLO` series, indicating whether to enable `Efficient NMS Plugin` to replace `TRTBatchedNMS plugin` in `MMDeploy` when exporting `onnx`.

You can refer to the official [efficient NMS plugins](https://github.com/NVIDIA/TensorRT/blob/main/plugin/efficientNMSPlugin/README.md) by `TensorRT` for more details.

Note: this out-of-box feature is **only available in TensorRT>=8.0**, no need to compile it by yourself.

### Configuration for Dynamic Inputs

#### 1. Model Config

When you deploy a dynamic input model, you don't need to modify any model configuration files but the deployment configuration files.

#### 2. Deployment Config

To deploy the `YOLOv5` in MMYOLO to `ONNXRuntime`, please refer to the [`detection_onnxruntime_dynamic.py`](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy/detection_onnxruntime_dynamic.py).

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

`backend_config` indicates the backend with `type='onnxruntime'`. Other parameters stay the same as the static input section.

To deploy the `YOLOv5` to `TensorRT`, please refer to the [`detection_tensorrt_dynamic-192x192-960x960.py`](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy/detection_tensorrt_dynamic-192x192-960x960.py).

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

`backend_config` indicates the backend with `type='tensorrt'`. Since the dynamic and static inputs are different in `TensorRT`, please check the details at [TensorRT dynamic input official introduction](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/developer-guide/index.html#work_dynamic_shapes).

`TensorRT` deployment requires you to specify `min_shape`, `opt_shape` , and `max_shape`. `TensorRT` limits the size of the input image between `min_shape` and `max_shape`.

`min_shape` is the minimum size of the input image. `opt_shape` is the common size of the input image, inference performance is best under this size. `max_shape` is the maximum size of the input image.

`use_efficientnms` configuration is the same as the `TensorRT` static input configuration in the previous section.

### INT8 Quantization Support

Note: Int8 quantization support will soon be released.

## How to Convert Model

### Usage

#### Deploy with MMDeploy Tools

Set the root directory of `MMDeploy` as an env parameter `MMDEPLOY_DIR` using `export MMDEPLOY_DIR=/the/root/path/of/MMDeploy` command.

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

### Parameter Description

- `deploy_cfg`: set the deployment config path of MMDeploy for the model, including the type of inference framework, whether quantize, whether the input shape is dynamic, etc. There may be a reference relationship between configuration files, e.g. `configs/deploy/detection_onnxruntime_static.py`
- `model_cfg`: set the MMYOLO model config path, e.g. `configs/deploy/model/yolov5_s-deploy.py`, regardless of the path to MMDeploy
- `checkpoint`: set the torch model path. It can start with `http/https`, more details are available in `mmengine.fileio` apis
- `img`: set the path to the image or point cloud file used for testing during model conversion
- `--test-img`: set the image file that used to test model. If not specified, it will be set to `None`
- `--work-dir`: set the work directory that used to save logs and models
- `--calib-dataset-cfg`: use for calibration only for INT8 mode. If not specified, it will be set to None and use “val” dataset in model config for calibration
- `--device`: set the device used for model conversion. The default is `cpu`, for TensorRT used `cuda:0`
- `--log-level`: set log level which in `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. If not specified, it will be set to `INFO`
- `--show`: show the result on screen or not
- `--dump-info`: output SDK information or not

#### Deploy with MMDeploy API

Suppose the working directory is the root path of mmyolo. Take [YoloV5](https://github.com/open-mmlab/mmyolo/blob/main/configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py) model as an example. You can download its checkpoint from [here](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth), and then convert it to onnx model as follows:

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

## Model specification

Before moving on to model inference chapter, let's know more about the converted result structure which is very important for model inference. It is saved in the directory specified with `--wodk_dir`.

The converted results are saved in the working directory `mmdeploy_models/mmyolo/onnx` in the previous example. It includes:

```
mmdeploy_models/mmyolo/onnx
├── deploy.json
├── detail.json
├── end2end.onnx
└── pipeline.json
```

in which,

- **end2end.onnx**: backend model which can be inferred by ONNX Runtime
- ***xxx*.json**: the necessary information for mmdeploy SDK

The whole package **mmdeploy_models/mmyolo/onnx** is defined as **mmdeploy SDK model**, i.e., **mmdeploy SDK model** includes both backend model and inference meta information.

## Model inference

### Backend model inference

Take the previous converted `end2end.onnx` model as an example, you can use the following code to inference the model and visualize the results.

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

With the above code, you can find the inference result `output_detection.png` in `work_dir`.

### SDK model inference

You can also perform SDK model inference like following,

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

Besides python API, mmdeploy SDK also provides other FFI (Foreign Function Interface), such as C, C++, C#, Java and so on. You can learn their usage from [demos](https://github.com/open-mmlab/mmdeploy/tree/main/demo).

## How to Evaluate Model

### Usage

After the model is converted to your backend, you can use `${MMDEPLOY_DIR}/tools/test.py` to evaluate the performance.

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

### Parameter Description

- `deploy_cfg`: set the deployment config file path.
- `model_cfg`: set the MMYOLO model config file path.
- `--model`: set the converted model. For example, if we exported a TensorRT model, we need to pass in the file path with the suffix ".engine".
- `--device`: indicate the device to run the model. Note that some backends limit the running devices. For example, TensorRT must run on CUDA.
- `--work-dir`: the directory to save the file containing evaluation metrics.
- `--cfg-options`: pass in additional configs, which will override the current deployment configs.
- `--show`: show the evaluation result on screen or not.
- `--show-dir`: save the evaluation result to this directory, valid only when specified.
- `--interval`: set the display interval between each two evaluation results.
- `--wait-time`: set the display time of each window.
- `--log2file`: log evaluation results and speed to file.
- `--speed-test`: test the inference speed or not.
- `--warmup`: warm up before speed test or not, works only when `speed-test` is specified.
- `--log-interval`: the interval between each log, works only when `speed-test` is specified.
- `--batch-size`: set the batch size for inference, which will override the `samples_per_gpu` in data config. The default value is `1`, however, not every model supports `batch_size > 1`.
- `--uri`: Remote ipv4:port or ipv6:port for inference on edge device.

Note: other parameters in `${MMDEPLOY_DIR}/tools/test.py` are used for speed test, they will not affect the evaluation results.
