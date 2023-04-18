# YOLOv5 部署全流程说明

请先参考 [`部署必备指南`](./mmdeploy_guide.md) 了解部署配置文件等相关信息。

## 模型训练和测试

模型训练和测试请参考 [YOLOv5 从入门到部署全流程](./mmdeploy_yolov5.md) 。

## 准备 MMDeploy 运行环境

安装 `MMDeploy` 请参考 [`源码手动安装`](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/docs/zh_cn/01-how-to-build/build_from_source.md) ，选择您所使用的平台编译 `MMDeploy` 和自定义算子。

*注意！* 如果环境安装有问题，可以查看 [`MMDeploy FAQ`](https://github.com/open-mmlab/mmdeploy/blob/dev-1.x/docs/zh_cn/faq.md) 或者在 `issuse` 中提出您的问题。

## 准备模型配置文件

本例将以基于 `coco` 数据集预训练的 `YOLOv5` 配置和权重进行部署的全流程讲解，包括静态/动态输入模型导出和推理，`TensorRT` / `ONNXRuntime` 两种后端部署和测试。

### 静态输入配置

#### (1) 模型配置文件

当您需要部署静态输入模型时，您应该确保模型的输入尺寸是固定的，比如在测试流程或测试数据集加载时输入尺寸为 `640x640`。

您可以查看 [`yolov5_s-static.py`](https://github.com/open-mmlab/mmyolo/tree/main/configs/deploy/model/yolov5_s-static.py) 中测试流程或测试数据集加载部分，如下所示：

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

由于 `yolov5` 在测试时会开启 `allow_scale_up` 和 `use_mini_pad` 改变输入图像的尺寸来取得更高的精度，但是会给部署静态输入模型造成输入尺寸不匹配的问题。

该配置相比与原始配置文件进行了如下修改:

- 关闭 `test_pipline` 中改变尺寸相关的配置，如 `LetterResize` 中 `allow_scale_up=False` 和 `use_mini_pad=False` 。
- 关闭 `test_dataloader` 中 `batch shapes` 策略，即 `batch_shapes_cfg=None` 。

#### (2) 部署配置文件

当您部署在 `ONNXRuntime` 时，您可以查看 [`detection_onnxruntime_static.py`](https://github.com/open-mmlab/mmyolo/tree/main/configs/deploy/detection_onnxruntime_static.py) ，如下所示：

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

默认配置中的 `post_processing` 后处理参数是当前模型与 `pytorch` 模型精度对齐的配置，若您需要修改相关参数，可以参考 [`部署必备指南`](./mmdeploy_guide.md) 的详细介绍。

当您部署在 `TensorRT` 时，您可以查看 [`detection_tensorrt_static-640x640.py`](https://github.com/open-mmlab/mmyolo/tree/main/configs/deploy/detection_tensorrt_static-640x640.py) ，如下所示：

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

本例使用了默认的输入尺寸 `input_shape=(640, 640)` ，构建网络以 `fp32` 模式即 `fp16_mode=False`，并且默认构建 `TensorRT` 构建引擎所使用的显存 `max_workspace_size=1 << 30` 即最大为 `1GB` 显存。

### 动态输入配置

#### (1) 模型配置文件

当您需要部署动态输入模型时，模型的输入可以为任意尺寸(`TensorRT` 会限制最小和最大输入尺寸)，因此使用默认的 [`yolov5_s-v61_syncbn_8xb16-300e_coco.py`](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py) 模型配置文件即可，其中数据处理和数据集加载器部分如下所示：

```python
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
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

其中 `LetterResize` 类初始化传入了 `allow_scale_up=False` 控制输入的小图像是否上采样，同时默认 `use_mini_pad=False` 关闭了图片最小填充策略，`val_dataloader['dataset']`中传入了 `batch_shapes_cfg=batch_shapes_cfg`，即按照 `batch` 内的输入尺寸进行最小填充。上述策略会改变输入图像的尺寸，因此动态输入模型在测试时会按照上述数据集加载器动态输入。

#### (2) 部署配置文件

当您部署在 `ONNXRuntime` 时，您可以查看 [`detection_onnxruntime_dynamic.py`](https://github.com/open-mmlab/mmyolo/blob/main/configs/deploy/detection_onnxruntime_dynamic.py) ，如下所示：

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

与静态输入配置仅有 `_base_ = ['./base_dynamic.py']` 不同，动态输入会额外继承 `dynamic_axes` 属性。其他配置与静态输入配置相同。

当您部署在 `TensorRT` 时，您可以查看 [`detection_tensorrt_dynamic-192x192-960x960.py`](https://github.com/open-mmlab/mmyolo/tree/main/configs/deploy/detection_tensorrt_dynamic-192x192-960x960.py) ，如下所示：

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

本例构建网络以 `fp32` 模式即 `fp16_mode=False`，构建 `TensorRT` 构建引擎所使用的显存 `max_workspace_size=1 << 30` 即最大为 `1GB` 显存。

同时默认配置 `min_shape=[1, 3, 192, 192]`，`opt_shape=[1, 3, 640, 640]` ，`max_shape=[1, 3, 960, 960]` ，意为该模型所能接受的输入尺寸最小为 `192x192` ，最大为 `960x960`，最常见尺寸为 `640x640`。

当您部署自己的模型时，需要根据您的输入图像尺寸进行调整。

## 模型转换

本教程所使用的 `MMDeploy` 根目录为 `/home/openmmlab/dev/mmdeploy`，请注意修改为您的 `MMDeploy` 目录。
预训练权重下载于 [yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth) ，保存在本地的 `/home/openmmlab/dev/mmdeploy/yolov5s.pth`。

```shell
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth -O /home/openmmlab/dev/mmdeploy/yolov5s.pth
```

命令行执行以下命令配置相关路径：

```shell
export MMDEPLOY_DIR=/home/openmmlab/dev/mmdeploy
export PATH_TO_CHECKPOINTS=/home/openmmlab/dev/mmdeploy/yolov5s.pth
```

### YOLOv5 静态输入模型导出

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

```shell
python3 ${MMDEPLOY_DIR}/tools/deploy.py \
    configs/deploy/detection_tensorrt_static-640x640.py \
    configs/deploy/model/yolov5_s-static.py \
    ${PATH_TO_CHECKPOINTS} \
    demo/demo.jpg \
    --work-dir work_dir \
    --show \
    --device cuda:0
```

### YOLOv5 动态输入模型导出

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

当您使用上述命令转换模型时，您将会在 `work_dir` 文件夹下发现以下文件：

![image](https://user-images.githubusercontent.com/92794867/199377596-605c3493-c1e0-435d-bc97-2e46846ac87d.png)

或者

![image](https://user-images.githubusercontent.com/92794867/199377848-a771f9c5-6bd6-49a1-9f58-e7e7b96c800f.png)

在导出 `onnxruntime`模型后，您将得到图1的三个文件，其中 `end2end.onnx` 表示导出的`onnxruntime`模型。

在导出 `TensorRT`模型后，您将得到图2的四个文件，其中 `end2end.onnx` 表示导出的中间模型，`MMDeploy`利用该模型自动继续转换获得 `end2end.engine` 模型用于 `TensorRT `部署。

## 模型评测

当您转换模型成功后，可以使用 `${MMDEPLOY_DIR}/tools/test.py` 工具对转换后的模型进行评测。下面是对 `ONNXRuntime` 和 `TensorRT` 静态模型的评测，动态模型评测修改传入模型配置即可。

### ONNXRuntime

```shell
python3 ${MMDEPLOY_DIR}/tools/test.py \
        configs/deploy/detection_onnxruntime_static.py \
        configs/deploy/model/yolov5_s-static.py \
        --model work_dir/end2end.onnx  \
        --device cpu \
        --work-dir work_dir
```

执行完成您将看到命令行输出检测结果指标如下：

![image](https://user-images.githubusercontent.com/92794867/199380483-cf8d867b-7309-4994-938a-f743f4cada77.png)

### TensorRT

**注意**： TensorRT 需要执行设备是 `cuda`

```shell
python3 ${MMDEPLOY_DIR}/tools/test.py \
        configs/deploy/detection_tensorrt_static-640x640.py \
        configs/deploy/model/yolov5_s-static.py \
        --model work_dir/end2end.engine  \
        --device cuda:0 \
        --work-dir work_dir
```

执行完成您将看到命令行输出检测结果指标如下：

![image](https://user-images.githubusercontent.com/92794867/199380370-da15cfca-2723-4e5b-b6cf-0afb5f44a66a.png)

**未来我们将会支持模型测速等更加实用的脚本**

# 使用 Docker 部署测试

`MMYOLO` 提供了一个 [`Dockerfile`](https://github.com/open-mmlab/mmyolo/blob/main/docker/Dockerfile_deployment) 用于构建镜像。请确保您的 `docker` 版本大于等于 `19.03`。

温馨提示；国内用户建议取消掉 [`Dockerfile`](https://github.com/open-mmlab/mmyolo/blob/main/docker/Dockerfile_deployment) 里面 `Optional` 后两行的注释，可以获得火箭一般的下载提速：

```dockerfile
# (Optional)
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

构建命令：

```bash
# build an image with PyTorch 1.12, CUDA 11.6, TensorRT 8.2.4 ONNXRuntime 1.8.1
docker build -f docker/Dockerfile_deployment -t mmyolo:v1 .
```

用以下命令运行 Docker 镜像：

```bash
export DATA_DIR=/path/to/your/dataset
docker run --gpus all --shm-size=8g -it --name mmyolo -v ${DATA_DIR}:/openmmlab/mmyolo/data/coco mmyolo:v1
```

`DATA_DIR` 是 COCO 数据的路径。

复制以下脚本到 `docker` 容器 `/openmmlab/mmyolo/script.sh`:

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

在 `/openmmlab/mmyolo` 下运行：

```bash
sh script.sh
```

脚本会自动下载 `MMYOLO` 的 `YOLOv5` 预训练权重并使用 `MMDeploy` 进行模型转换和测试。您将会看到以下输出：

- TensorRT：

  ![image](https://user-images.githubusercontent.com/92794867/199657349-1bad9196-c00b-4a65-84f5-80f51e65a2bd.png)

- ONNXRuntime：

  ![image](https://user-images.githubusercontent.com/92794867/199657283-95412e84-3ba4-463f-b4b2-4bf52ec4acbd.png)

可以看到，经过 `MMDeploy` 部署的模型与 [MMYOLO-YOLOv5](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov5#results-and-models) 的 mAP-37.7 差距在 1% 以内。

如果您需要测试您的模型推理速度，可以使用以下命令：

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

## 模型推理

TODO
