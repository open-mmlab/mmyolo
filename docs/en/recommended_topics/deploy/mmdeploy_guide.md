# Basic Deployment Guide

## Introduction of MMDeploy

MMDeploy is an open-source deep learning model deployment toolset. It is a part of the [OpenMMLab](https://openmmlab.com/) project, and provides **a unified experience of exporting different models** to various platforms and devices of the OpenMMLab series libraries. Using MMDeploy, developers can easily export the specific compiled SDK they need from the training result, which saves a lot of effort.

More detailed introduction and guides can be found [here](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/docs/en/get_started.md)

## Supported Algorithms

Currently our deployment kit supports on the following models and backends:

| Model  | Task            | OnnxRuntime | TensorRT |                              Model config                               |
| :----- | :-------------- | :---------: | :------: | :---------------------------------------------------------------------: |
| YOLOv5 | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov5) |
| YOLOv6 | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov6) |
| YOLOX  | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolox)  |
| RTMDet | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet) |

Note: ncnn and other inference backends support are coming soon.

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
- `iou_threshold`: set the `iou` threshold for removing duplicates in `nums`
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

Here we still use the `YOLOv5` in MMYOLO as the example. We can use [`detection_onnxruntime_static.py`](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy/detection_onnxruntime_static.py) as the config to deploy \`YOLOv5\` to \`ONNXRuntim\` with static inputs.

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

`backend_config` indices the backend with `type=‘tensorrt’`.

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

## How to Evaluate Model

### Usage

After the model is converted to your backend, you can use `${MMDEPLOY_DIR}/tools/test.py` to evaluate the performance.

```shell
python3 ${MMDEPLOY_DIR}/tools/test.py \
    ${DEPLOY_CFG} \
    ${MODEL_CFG} \
    --model ${BACKEND_MODEL_FILES} \
    [--out ${OUTPUT_PKL_FILE}] \
    [--format-only] \
    [--metrics ${METRICS}] \
    [--show] \
    [--show-dir ${OUTPUT_IMAGE_DIR}] \
    [--show-score-thr ${SHOW_SCORE_THR}] \
    --device ${DEVICE} \
    [--cfg-options ${CFG_OPTIONS}] \
    [--metric-options ${METRIC_OPTIONS}]
    [--log2file work_dirs/output.txt]
    [--batch-size ${BATCH_SIZE}]
    [--speed-test] \
    [--warmup ${WARM_UP}] \
    [--log-interval ${LOG_INTERVERL}]
```

### Parameter Description

- `deploy_cfg`: set the deployment config file path
- `model_cfg`: set the MMYOLO model config file path
- `--model`: set the converted model. For example, if we exported a TensorRT model, we need to pass in the file path with the suffix ".engine"
- `--out`: save the output result in pickle format, use only when you need it
- `--format-only`: format the output without evaluating it. It is useful when you want to format the result into a specific format and submit it to a test server
- `--metrics`: use the specific metric supported in MMYOLO to evaluate, such as "proposal" in COCO format data.
- `--show`: show the evaluation result on screen or not
- `--show-dir`: save the evaluation result to this directory, valid only when specified
- `--show-score-thr`: show the threshold for the detected bboxes or not
- `--device`: indicate the device to run the model. Note that some backends limit the running devices. For example, TensorRT must run on CUDA
- `--cfg-options`: pass in additional configs, which will override the current deployment configs
- `--metric-options`: add custom options for metrics. The key-value pair format in xxx=yyy will be the kwargs of the dataset.evaluate() method
- `--log2file`: save the evaluation results (with the speed) to a file
- `--batch-size`: set the batch size for inference, which will override the `samples_per_gpu` in data config. The default value is `1`, however, not every model supports `batch_size > 1`
- `--speed-test`: test the inference speed or not
- `--warmup`: warm up before speed test or not, works only when `speed-test` is specified
- `--log-interval`: set the interval between each log, works only when `speed-test` is specified

Note: other parameters in `${MMDEPLOY_DIR}/tools/test.py` are used for speed test, they will not affect the evaluation results.
