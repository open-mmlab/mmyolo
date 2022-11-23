# MMYOLO 模型 ONNX 转换

## 环境依赖

- [onnx](https://github.com/onnx/onnx)

  ```shell
  pip install onnx
  ```

  [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) (可选，用于简化模型)

  ```shell
  pip install onnx-simplifier
  ```

## 使用方法

[模型导出脚本](./projects/easydeploy/tools/export.py)用于将 `MMYOLO` 模型转换为 `onnx` 。

### 参数介绍:

- `config` : 构建模型使用的配置文件，如 [`yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py`](./configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py) 。
- `checkpoint` : 训练得到的权重文件，如 `yolov5s.pth` 。
- `--work-dir` : 转换后的模型保存路径。
- `--img-size`: 转换模型时输入的尺寸，如 `640 640`。
- `--batch-size`: 转换后的模型输入 `batch size` 。
- `--device`: 转换模型使用的设备，默认为 `cuda:0`。
- `--simplify`: 是否简化导出的 `onnx` 模型，需要安装 [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)，默认关闭。
- `--opset`: 指定导出 `onnx` 的 `opset`，默认为 `11` 。
- `--backend`: 指定导出 `onnx` 用于的后端 id，`ONNXRuntime`: `1`, `TensorRT8`: `2`, `TensorRT7`: `3`，默认为`1`即 `ONNXRuntime`。
- `--pre-topk`: 指定导出 `onnx` 的后处理筛选候选框个数阈值，默认为 `1000`。
- `--keep-topk`: 指定导出 `onnx` 的非极大值抑制输出的候选框个数阈值，默认为 `100`。
- `--iou-threshold`: 非极大值抑制中过滤重复候选框的 `iou` 阈值，默认为 `0.65`。
- `--score-threshold`: 非极大值抑制中过滤候选框得分的阈值，默认为 `0.25`。

例子:

```shell
python ./projects/easydeploy/tools/export.py \
	configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
	yolov5s.pth \
	--work-dir work_dir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 1 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```

然后利用后端支持的工具如 `TensorRT` 读取 `onnx` 再次转换为后端支持的模型格式如 `.engine/.plan` 等
