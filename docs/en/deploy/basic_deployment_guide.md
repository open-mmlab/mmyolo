# Deployment Prerequisites Tutorial

## MMDeploy Introduce

MMDeploy is [OpenMMLab](https://openmmlab.com/) model deployment toolkit, **providing a unified deployment experience for each algorithm library**. Based on MMDeploy, developers can easily generate SDKs for specified hardware from training repo, saving a lot of adaptation time.

More information and instructions see https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/docs/zh_cn/get_started.md

## Algorithm Support List

Currently supported model-backend combinations:

| Model  | Task            | OnnxRuntime | TensorRT |                              Model config                               |
| :----- | :-------------- | :---------: | :------: | :---------------------------------------------------------------------: |
| YOLOv5 | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov5) |
| YOLOv6 | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov6) |
| YOLOX  | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolox)  |
| RTMDet | ObjectDetection |      Y      |    Y     | [config](https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet) |

ncnn and other backend support will be supported later.

## Deployment Related Configuration Description in MMYOLO

All deployment configuration files are in directory [`configs/deploy`](configs/deploy).

You can deploy static input or dynamic input models, so you need to modify the data processing flow related to this in the model configuration file.

MMDeploy integrates post-processing into user-defined operators, so you can modify parameter `post_processing` in `codebase_config` to adjust post-processing strategies. The parameters are described as follows:

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

- `score_threshold`：the category score threshold for filtering candidate boxes before `nms`.
- `confidence_threshold`：the confidence score threshold for filtering candidate boxes before `nms`.
- `iou_threshold`：the 'iou' threshold for removing duplicate boxes in `nms`.
- `max_output_boxes_per_class`：the maximum number of output boxes per category.
- `pre_top_k`：the number of candidate boxes that sort the candidate box scores and then fix before `nms`.
- `keep_top_k`：the number of candidate boxes that the `nms` algorithm finally outputs.
- `background_label_id`：MMYOLO has no background category information, just set it to `-1`.

### Static Input Configuration

#### (1) Introduction to Model Configuration Files

Taking the `YOLOv5` model configuration in MMYOLO as an example, the following is an introduction to the parameters of the model configuration file used for deployment.

```python
_base_ = '../../yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
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

`_base_ = '../../yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'` inherits the configuration for building the model during training.

`test_pipeline` is the process of processing input images for deployment,
`LetterResize` controls the size of the input image, while limiting the input size that the exported model can accept.

`test_dataloader` builds the data loader configuration for deployment,
`batch_shapes_cfg` controls whether to enable the `batch_shapes` strategy, for details, please refer to [yolov5 configuration file description](https://github.com/open-mmlab/mmyolo/blob/main/docs/zh_cn/user_guides/config.md) .

#### (2) Introduction to Deployment Files

Taking the `YOLOv5` deployment configuration in `MMYOLO` as an example, the following is an introduction to the configuration file parameters.

`ONNXRuntime` deploys `YOLOv5` can use [`detection_onnxruntime_static.py`](configs/deploy/detection_onnxruntime_static.py) configuration.

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

`backend_config` specifies the deployment backend `type='onnxruntime'`, other information can refer to the third section.

`TensorRT` deploys `YOLOv5` can use [`detection_tensorrt_static-640x640.py`](config/deploy/detection_tensorrt_static-640x640.py) configuration.

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

`backend_config` specifies the deployment backend `type=‘tensorrt’`.

Different from `ONNXRuntime` deployment configuration, `TensorRT` needs to specify the input image size and the parameters required to build the engine file, including:

- `onnx_config` specifies `input_shape=(640, 640)`
- `backend_config['common_config']` has `fp16_mode=False` and `max_workspace_size=1 << 30`, `fp16_mode` indicates whether to build the engine in the parameter format of `fp16`,`max_workspace_size` indicates the maximum memory of the current `gpu` device, and the unit is `GB`.For detailed configuration of `fp16`, please refer to [`detection_tensorrt-fp16_static-640x640.py`](configs/deploy/detection_tensorrt-fp16_static-640x640.py)
- `backend_config['model_inputs']['input_shapes']['input']` needs `min_shape` /`opt_shape`/`max_shape` the corresponding value should remain the same under static input, the default is `[1, 3, 640, 640]`.

`use_efficientnms` is a new configuration introduced by the `MMYOLO` series, indicating whether to enable `Efficient NMS Plugin` to replace `TRTBatchedNMS plugin` in `MMDeploy` when exporting `onnx`.

You can refer to `TensorRT` official implementation of [Efficient NMS Plugin](https://github.com/NVIDIA/TensorRT/blob/main/plugin/efficientNMSPlugin/README.md) for more details.

**Note**, this feature is only available in TensorRT >= 8.0 version, no need to compile it out of the box.

### Dynamic Input Configuration

#### (1) Introduction to Model Configuration Files

When you deploy a static input model, you don't need to modify any model configuration files, only the deployment configuration file.

#### (2) Introduction to Deployment Files

`ONNXRuntime` deploys `YOLOv5` can use  [`detection_onnxruntime_dynamic.py`](configs/deploy/detection_onnxruntime_dynamic.py)  configuration.

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

`backend_config` specifies the backend `type='onnxruntime'`, and other configurations are the same as the static input model deployed in ONNXRuntime in the previous section.

`TensorRT` deploys `YOLOv5` can use  [`detection_tensorrt_dynamic-192x192-960x960.py`](config/deploy/detection_tensorrt_dynamic-192x192-960x960.py) configuration.

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

`backend_config` specifies the backend `type='tensorrt'`, due to `TensorRT` dynamic input is different from static input, you can learn more about dynamic input by visiting [TensorRT dynamic input official introduction](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/developer-guide/index.html#work_dynamic_shapes).

`TensorRT` deploys requires configuration `min_shape`, `opt_shape`, `max_shape` , `TensorRT` limit the size of the input image between `min_shape` and `max_shape`.

`min_shape` is the minimum size of the input image, `opt_shape` is the common size for input images, inference performance is best at this size, `max_shape` is the maximum size of the input image.

`use_efficientnms` configuration is the same as the `TensorRT` static input configuration in the previous section.

### INT8 Quantization Configuration

!!! Tutorial on deploying TensorRT INT8 models coming soon !!!

## Model Conversion

### Instructions

Set the `MMDeploy` root directory to the environment variable `MMDEPLOY_DIR`, for example`export MMDEPLOY_DIR=/the/root/path/of/MMDeploy`

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

- `deploy_cfg` : mmdeploy deployment configuration for this model, including the type of inference framework, whether it is quantized, whether the input shape is dynamic, etc. There may be reference relationships between configuration files, `configs/deploy/detection_onnxruntime_static.py` is an example.
- `model_cfg` : model configuration of MMYOLO algorithm library, such as `configs/deploy/model/yolov5_s-deploy.py`, has nothing to do with the path of mmdeploy.
- `checkpoint` : torch model path. It can start with http/https, see the implementation of `mmcv.FileClient` for details. .
- `img` : the path to the image file used for testing when converting the model.
- `--test-img` : the path to the image file used to test the model. Default is set to `None`.
- `--work-dir` : working directory, used to save log and model files.
- `--calib-dataset-cfg` : this parameter is only valid in int8 mode and is used to calibrate the dataset configuration file. If no parameter is passed in int8 mode, the 'val' data set in the model configuration file will be used for calibration automatically.
- `--device` : device for model conversion. The default is `cpu`, and `cuda:0` can be used for trt.
- `--log-level` : set the log level, options include `'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'`. The default is `INFO`.
- `--show` : whether to display the detection results.
- `--dump-info` : whether to output SDK information.

## Model Evaluation

After you convert the PyTorch model to the model supported by the backend, you may need to verify the accuracy of the model, use `${MMDEPLOY_DIR}/tools/test.py`

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

### 参数描述

- `deploy_cfg`: deployment configuration file.
- `model_cfg`: MMYOLO model configuration file.
- `--model`: exported backend model. For example, if we exported the TensorRT model, we need to pass in the file path with the suffix ".engine".
- `--out`: save the output in pickle format, only enabled if you pass this parameter.
- `--format-only`: whether to format the output without evaluating it. It is useful when you want to format the result into a specific format and submit it to the test server.
- `--metrics`: metrics for evaluating models defined in MMYOLO, such as "proposal" in COCO annotation format.
- `--show`: whether to show the evaluation results on the screen.
- `--show-dir`: directory to save evaluation results. (Results are saved only if this parameter is given).
- `--show-score-thr`: determines whether to show the threshold for detecting bounding boxes.
- `--device`: the device to run the model on. Note that some backends limit devices. For example, TensorRT must run on cuda.
- `--cfg-options`: pass in additional configuration that will override the current deployment configuration.
- `--metric-options`: custom options for metrics. The key-value pair format in xxx=yyy will be the kwargs of the dataset.evaluate() function.
- `--log2file`: log evaluation results (and velocities) to a file.
- `--batch-size`: batch size for inference, it will override `samples_per_gpu` in data config. Defaults to `1`. Note that not all models support `batch_size > 1`.
- `--speed-test`: whether to enable speed test.
- `--warmup`: to warm up before calculating the inference time, you need to enable `speed-test` first.
- `--log-interval`: the interval between each log, you need to set `speed-test` first.

Note: Additional parameters in `${MMDEPLOY_DIR}/tools/test.py` are used for speed tests. They do not affect the evaluation.
