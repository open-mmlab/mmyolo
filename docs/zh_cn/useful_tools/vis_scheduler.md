# 可视化优化器参数策略

`tools/analysis_tools/vis_scheduler.py` 旨在帮助用户检查优化器的超参数调度器（无需训练），支持学习率（learning rate）、动量（momentum）和权值衰减（weight decay）。

```shell
python tools/analysis_tools/vis_scheduler.py \
    ${CONFIG_FILE} \
    [-p, --parameter ${PARAMETER_NAME}] \
    [-d, --dataset-size ${DATASET_SIZE}] \
    [-n, --ngpus ${NUM_GPUs}] \
    [-o, --out-dir ${OUT_DIR}] \
    [--title ${TITLE}] \
    [--style ${STYLE}] \
    [--window-size ${WINDOW_SIZE}] \
    [--cfg-options]
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- **`-p, parameter`**: 可视化参数名，只能为 `["lr", "momentum", "wd"]` 之一， 默认为 `"lr"`.
- **`-d, --dataset-size`**: 数据集的大小。如果指定，`DATASETS.build` 将被跳过并使用这个数值作为数据集大小，默认使用 `DATASETS.build` 所得数据集的大小。
- **`-n, --ngpus`**: 使用 GPU 的数量, 默认为1。
- **`-o, --out-dir`**: 保存的可视化图片的文件夹路径，默认不保存。
- `--title`: 可视化图片的标题，默认为配置文件名。
- `--style`: 可视化图片的风格，默认为 `whitegrid`。
- `--window-size`: 可视化窗口大小，如果没有指定，默认为 `12*7`。如果需要指定，按照格式 `'W*H'`。
- `--cfg-options`: 对配置文件的修改，参考[学习配置文件](../tutorials/config.md)。

```{note}
部分数据集在解析标注阶段比较耗时，推荐直接将 `-d, dataset-size` 指定数据集的大小，以节约时间。
```

你可以使用如下命令来绘制配置文件 `configs/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py` 将会使用的学习率变化曲线：

```shell
python tools/analysis_tools/vis_scheduler.py \
    configs/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py \
    --dataset-size 118287 \
    --ngpus 8 \
    --out-dir ./output
```

<div align=center><img src="https://user-images.githubusercontent.com/27466624/213091635-d322d2b3-6e28-4755-b871-ef0a89a67a6b.jpg" style=" width: auto; height: 40%; "></div>
