# YOLOv5 Deployment

Please refer to [`Deployment Essentials Guide`](./部署必备指南.md) to learn about deployment configuration files and other related information.

## Model Training and Testing

For model training and testing, please refer to [`From getting started to deployment with YOLOv5`](docs/en/user_guides/yolov5_tutorial.md).

## Prepare the MMDeploy Runtime Environment

Install `MMDeploy` Please refer to [`source code manual installation`](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/docs/zh_cn/01-how-to-build/build_from_source.md), Select the platform you used to compile `MMDeploy` and custom operators.

*Notice!* If there is a problem with the environment installation, you can check [`MMDeploy FAQ`](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/docs/zh_cn/faq.md) or ask your question in `issuse`.

## Prepare Model Configuration Files

This example will explain the whole process of deployment based on the `YOLOv5` configuration and weights pre-trained on the `coco` dataset, including static/dynamic input model export and inference, `TensorRT` / `ONNXRuntime` two back-end deployment and testing .

### Static Input Configuration

#### (1) Model Configuration File

When you need to deploy a static input model, you should ensure that the input size of the model is fixed, such as `640x640` when the test process or test dataset is loaded.

You can check the test process or test dataset loading part in [`yolov5_s-deploy.py`](configs/deploy/model/yolov5_s-deploy.py), as follows:

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

Compared with the original configuration file, this configuration has been modified as follows:

- Turn off size-related configurations in `test_pipline`, such as `allow_scale_up=False` and `use_mini_pad=False` in `LetterResize`.
- Turn off the `batch shapes` policy in `test_dataloader`, ie `batch_shapes_cfg=None`.

#### (2) Deployment Configuration File

When you deploy on `ONNXRuntime`, you can view [`detection_onnxruntime_static.py`](configs/deploy/detection_onnxruntime_static.py) , as follows：

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

The `post_processing` post-processing parameters in the default configuration are configurations that align the accuracy of the current model with the `pytorch` model. If you need to modify the relevant parameters, you can refer to [`Deployment Essentials Guide`](./部署必备指南) detailed introduction.

When you deploy on `TensorRT`, you can view [`detection_tensorrt_static-640x640.py`](config/deploy/detection_tensorrt_static-640x640.py), as follows:

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

This example uses the default input size `input_shape=(640, 640)`, builds the network in `fp32` mode, namely `fp16_mode=False`, and builds `TensorRT` by default. The video memory used by the construction engine `max_workspace_size=1 << 30` means a maximum of `1GB` video memory.

### Dynamic Input Configuration

#### (1) Model Configuration File

When you need to deploy a dynamic input model, the input of the model can be of any size (`TensorRT` will limit the minimum and maximum input size), so use the default [`yolov5_s-v61_syncbn_8xb16-300e_coco.py`](configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py) model configuration file, the data processing and dataset loader parts are as follows:

```python
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img='val2017/'),
        ann_file='annotations/instances_val2017.json',
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))
```

Among them, the `LetterResize` class is initialized with `allow_scale_up=False` to control whether the input small image is upsampled. At the same time, the default `use_mini_pad=False` turns off the minimum padding strategy of the image, and `val_dataloader['dataset']` is passed in` batch_shapes_cfg=batch_shapes_cfg`, that is, the minimum padding is performed according to the input size in `batch`. The above strategy will change the dimensions of the input image, so the dynamic input model will be dynamically input according to the above dataset loader when testing.

#### (2) Deployment Configuration File

When you deploy on `ONNXRuntime`, you can look at [`detection_onnxruntime_dynamic.py`](configs/deploy/detection_onnxruntime_dynamic.py) as follows:

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

Unlike static typing configuration which only has `_base_ = ['./base_dynamic.py']`, dynamic typing additionally inherits the `dynamic_axes` property. Other configurations are the same as static input configurations.

When you deploy on `TensorRT`, you can view [`detection_tensorrt_dynamic-192x192-960x960.py`](config/deploy/detection_tensorrt_dynamic-192x192-960x960.py) as follows:

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

In this example, the network is built in `fp32` mode, that is, `fp16_mode=False`, and the video memory used to build the `TensorRT` build engine is `max_workspace_size=1 << 30`, that is, the maximum video memory is `1GB`.

At the same time, the default configuration `min_shape=[1, 3, 192, 192]`, `opt_shape=[1, 3, 640, 640]`, `max_shape=[1, 3, 960, 960]`, which means that the model The minimum accepted input size is `192x192`, the largest is `960x960`, and the most common size is `640x640`.

When you deploy your own model, it needs to be adjusted to your input image dimensions.

## Model Conversion

The `MMDeploy` root directory used in this tutorial is `/home/openmmlab/dev/mmdeploy`, please pay attention to modify it to your `MMDeploy` directory.
Save it locally at `/home/openmmlab/dev/mmdeploy/yolov5s.pth`.

```shell
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth -O /home/openmmlab/dev/mmdeploy/yolov5s.pth
```

Execute the following commands on the command line to configure the relevant paths:

```shell
export MMDEPLOY_DIR=/home/openmmlab/dev/mmdeploy
export PATH_TO_CHECKPOINTS=/home/openmmlab/dev/mmdeploy/yolov5s.pth
```

### YOLOv5 Static Input Model Export

#### ONNXRuntime

```shell
python3 ${MMDEPLOY_DIR}/tools/deploy.py \
    configs/deploy/detection_onnxruntime_static.py \
    configs/deploy/model/yolov5_s-static.py \
    ${PATH_TO_CHECKPOINTS} \
    demo/demo.jpg \
    --work-dir work_dir \
    --show \
    --device cpu
```

#### TensorRT

```bash
python3 ${MMDEPLOY_DIR}/tools/deploy.py \
    configs/deploy/detection_tensorrt_static-640x640.py \
    configs/deploy/model/yolov5_s-static.py \
    ${PATH_TO_CHECKPOINTS} \
    demo/demo.jpg \
    --work-dir work_dir \
    --show \
    --device cuda:0
```

### YOLOv5 Dynamic Input Model Export

#### ONNXRuntime

```shell
python3 ${MMDEPLOY_DIR}/tools/deploy.py \
    configs/deploy/detection_onnxruntime_dynamic.py \
    configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py \
    ${PATH_TO_CHECKPOINTS} \
    demo/demo.jpg \
    --work-dir work_dir \
    --show \
    --device cpu
```

#### TensorRT

```shell
python3 ${MMDEPLOY_DIR}/tools/deploy.py \
    configs/deploy/detection_tensorrt_dynamic-192x192-960x960.py \
    configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py \
    ${PATH_TO_CHECKPOINTS} \
    demo/demo.jpg \
    --work-dir work_dir \
    --show \
    --device cuda:0
```

When you convert the model using the above command, you will find the following files under the `work_dir` folder:

![image](https://user-images.githubusercontent.com/92794867/199377596-605c3493-c1e0-435d-bc97-2e46846ac87d.png)

or

![image](https://user-images.githubusercontent.com/92794867/199377848-a771f9c5-6bd6-49a1-9f58-e7e7b96c800f.png)

After exporting the `onnxruntime` model, you will get three files as shown in Figure 1, where `end2end.onnx` represents the exported `onnxruntime` model.

After exporting the `TensorRT` model, you will get the four files in Figure 2, where `end2end.onnx` represents the exported intermediate model, `MMDeploy` uses this model to automatically continue to convert the `end2end.engine` model for `TensorRT `Deployment.

## Model Evaluation

After you convert the model successfully, you can use `${MMDEPLOY_DIR}/tools/test.py` tool to evaluate the converted model. The following is the evaluation of the static models of `ONNXRuntime` and `TensorRT`. For dynamic model evaluation, modify the configuration of the input model.

### ONNXRuntime

```shell
python3 ${MMDEPLOY_DIR}/tools/test.py \
        configs/deploy/detection_onnxruntime_static.py \
        configs/deploy/model/yolov5_s-static.py \
        --model work_dir/end2end.onnx  \
        --device cpu \
        --work-dir work_dir
```

After the execution is complete, you will see the command line output detection result indicators as follows:

![image](https://user-images.githubusercontent.com/92794867/199380483-cf8d867b-7309-4994-938a-f743f4cada77.png)

### TensorRT

**Note**: TensorRT needs to execute the device is `cuda`

```shell
python3 ${MMDEPLOY_DIR}/tools/test.py \
        configs/deploy/detection_tensorrt_static-640x640.py \
        configs/deploy/model/yolov5_s-static.py \
        --model work_dir/end2end.engine  \
        --device cuda:0 \
        --work-dir work_dir
```

After the execution is complete, you will see the command line output detection result indicators as follows:

![image](https://user-images.githubusercontent.com/92794867/199380370-da15cfca-2723-4e5b-b6cf-0afb5f44a66a.png)

**In the future, we will support more practical scripts such as model speed measurement**

# Deploy Testing with Docker

`MMYOLO` provides a [`Dockerfile`](docker/Dockerfile_deployment) for building images. Please make sure your `docker` version is greater than or equal to `19.03`.

Warm reminder; domestic users suggest uncommenting the last two lines of `Optional` in [`Dockerfile`](docker/Dockerfile_deployment) to get a rocket-like download speed:

```dockerfile
# (Optional)
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Build command:

```shell
# build an image with PyTorch 1.12, CUDA 11.6, TensorRT 8.2.4 ONNXRuntime 1.8.1
docker build -f docker/Dockerfile_deployment -t mmyolo:v1 .
```

Run the Docker image with the following command:

```shell
export DATA_DIR=/path/to/your/dataset
docker run --gpus all --shm-size=8g -it --name mmyolo -v ${DATA_DIR}:/openmmlab/mmyolo/data/coco mmyolo:v1
```

`DATA_DIR` is the path to the COCO data.

Copy the following script to `docker` container `/openmmlab/mmyolo/script.sh`:

```bash
#!/bin/bash
wget -q https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
  -O yolov5s.pth
export MMDEPLOY_DIR=/openmmlab/mmdeploy
export PATH_TO_CHECKPOINTS=/openmmlab/mmyolo/yolov5s.pth

python3 ${MMDEPLOY_DIR}/tools/deploy.py \
  configs/deploy/detection_tensorrt_static-640x640.py \
  configs/deploy/model/yolov5_s-static.py \
  ${PATH_TO_CHECKPOINTS} \
  demo/demo.jpg \
  --work-dir work_dir_trt \
  --device cuda:0

python3 ${MMDEPLOY_DIR}/tools/test.py \
  configs/deploy/detection_tensorrt_static-640x640.py \
  configs/deploy/model/yolov5_s-static.py \
  --model work_dir_trt/end2end.engine \
  --device cuda:0 \
  --work-dir work_dir_trt

python3 ${MMDEPLOY_DIR}/tools/deploy.py \
  configs/deploy/detection_onnxruntime_static.py \
  configs/deploy/model/yolov5_s-static.py \
  ${PATH_TO_CHECKPOINTS} \
  demo/demo.jpg \
  --work-dir work_dir_ort \
  --device cpu

python3 ${MMDEPLOY_DIR}/tools/test.py \
  configs/deploy/detection_onnxruntime_static.py \
  configs/deploy/model/yolov5_s-static.py \
  --model work_dir_ort/end2end.onnx \
  --device cpu \
  --work-dir work_dir_ort
```

Run under `/openmmlab/mmyolo`:

```shell
sh script.sh
```

The script will automatically download the `YOLOv5` pre-trained weights of `MMYOLO` and use `MMDeploy` for model conversion and testing. You will see the following output:

- TensorRT：

  ![image](https://user-images.githubusercontent.com/92794867/199657349-1bad9196-c00b-4a65-84f5-80f51e65a2bd.png)

- ONNXRuntime：

  ![image](https://user-images.githubusercontent.com/92794867/199657283-95412e84-3ba4-463f-b4b2-4bf52ec4acbd.png)

It can be seen that the model deployed by `MMDeploy` has a gap of 1% between the mAP-37.7 of [MMYOLO-YOLOv5](`https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov5`) within.

If you need to test your model inference speed, you can use the following command:

- TensorRT

```shell
python3 ${MMDEPLOY_DIR}/tools/profiler.py \
  configs/deploy/detection_tensorrt_static-640x640.py \
  configs/deploy/model/yolov5_s-static.py \
  data/coco/val2017 \
  --model work_dir_trt/end2end.engine \
  --device cuda:0
```

- ONNXRuntime

```shell
python3 ${MMDEPLOY_DIR}/tools/profiler.py \
  configs/deploy/detection_onnxruntime_static.py \
  configs/deploy/model/yolov5_s-static.py \
  data/coco/val2017 \
  --model work_dir_ort/end2end.onnx \
  --device cpu
```

## Model Inference

TODO
