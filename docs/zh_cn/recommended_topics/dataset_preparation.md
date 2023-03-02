# 数据集格式准备和说明

## DOTA 数据集

### 下载 DOTA 数据集

数据集可以从 DOTA 数据集的主页 [DOTA](https://captain-whu.github.io/DOTA/dataset.html)
或 [OpenDataLab](https://opendatalab.org.cn/DOTA_V1.0) 下载。

我们推荐使用 [OpenDataLab](https://opendatalab.org.cn/DOTA_V1.0) 下载，其中的文件夹结构已经按照需要排列好了，只需要解压即可，不需要费心去调整文件夹结构。

下载后解压数据集，并按如下文件夹结构存放。

```none
${DATA_ROOT}
├── train
│   ├── images
│   │   ├── P0000.png
│   │   ├── ...
│   ├── labelTxt-v1.0
│   │   ├── labelTxt
│   │   │   ├── P0000.txt
│   │   │   ├── ...
│   │   ├── trainset_reclabelTxt
│   │   │   ├── P0000.txt
│   │   │   ├── ...
├── val
│   ├── images
│   │   ├── P0003.png
│   │   ├── ...
│   ├── labelTxt-v1.0
│   │   ├── labelTxt
│   │   │   ├── P0003.txt
│   │   │   ├── ...
│   │   ├── valset_reclabelTxt
│   │   │   ├── P0003.txt
│   │   │   ├── ...
├── test
│   ├── images
│   │   ├── P0006.png
│   │   ├── ...

```

其中，以 `reclabelTxt` 为结尾的文件夹存放了水平检测框的标注，目前仅使用了 `labelTxt-v1.0` 中旋转框的标注。

### 数据集切片

我们提供了 `tools/dataset_converters/dota/dota_split.py` 脚本用于 DOTA 数据集的准备和切片。

```shell
python tools/dataset_converters/dota/dota_split.py \
    [--splt-config ${SPLIT_CONFIG}] \
    [--data-root ${DATA_ROOT}] \
    [--out-dir ${OUT_DIR}] \
    [--ann-subdir ${ANN_SUBDIR}] \
    [--phase ${DATASET_PHASE}] \
    [--nproc ${NPROC}] \
    [--save-ext ${SAVE_EXT}] \
    [--overwrite]
```

脚本依赖于 shapely 包，请先通过 `pip install shapely` 安装 shapely。

**参数说明**：

- `--splt-config` : 切片参数的配置文件。
- `--data-root`: DOTA 数据集的存放位置。
- `--out-dir`: 切片后的输出位置。
- `--ann-subdir`: 标注文件夹的名字。 默认为 `labelTxt-v1.0` 。
- `--phase`: 数据集的阶段。默认为 `trainval test` 。
- `--nproc`: 进程数量。 默认为 8 。
- `--save-ext`: 输出图像的扩展名，如置空则与原图保持一致。 默认为 `None` 。
- `--overwrite`: 如果目标文件夹已存在，是否允许覆盖。

基于 DOTA 数据集论文中提供的配置，我们提供了两种切片配置。

`./split_config/single_scale.json` 用于单尺度 `single-scale` 切片
`./split_config/multi_scale.json` 用于多尺度 `multi-scale` 切片

DOTA 数据集通常使用 `trainval` 集进行训练，然后使用 `test` 集进行在线验证，大多数论文提供的也是在线验证的精度。
如果你需要进行本地验证，可以准备 `train` 集和 `val` 集进行训练和测试。

示例:

使用单尺度切片配置准备 `trainval` 和 `test` 集

```shell
python tools/dataset_converters/dota/dota_split.py
    --split-config 'tools/dataset_converters/dota/split_config/single_scale.json'
    --data-root ${DATA_ROOT} \
    --out-dir ${OUT_DIR}
```

准备 DOTA-v1.5 数据集，它的标注文件夹名字是 `labelTxt-v1.5`

```shell
python tools/dataset_converters/dota/dota_split.py
    --split-config 'tools/dataset_converters/dota/split_config/single_scale.json'
    --data-root ${DATA_ROOT} \
    --out-dir ${OUT_DIR} \
    --ann-subdir 'labelTxt-v1.5'
```

使用单尺度切片配置准备 `train` 和 `val` 集

```shell
python tools/dataset_converters/dota/dota_split.py
    --split-config 'tools/dataset_converters/dota/split_config/single_scale.json'
    --data-root ${DATA_ROOT} \
    --phase train val \
    --out-dir ${OUT_DIR}
```

使用多尺度切片配置准备 `trainval` 和 `test` 集

```shell
python tools/dataset_converters/dota/dota_split.py
    --split-config 'tools/dataset_converters/dota/split_config/multi_scale.json'
    --data-root ${DATA_ROOT} \
    --out-dir ${OUT_DIR}
```

在运行完成后，输出的结构如下：

```none
${OUT_DIR}
├── trainval
│   ├── images
│   │   ├── P0000__1024__0___0.png
│   │   ├── ...
│   ├── annfiles
│   │   ├── P0000__1024__0___0.txt
│   │   ├── ...
├── test
│   ├── images
│   │   ├── P0006__1024__0___0.png
│   │   ├── ...
│   ├── annfiles
│   │   ├── P0006__1024__0___0.txt
│   │   ├── ...
```

此时将配置文件中的 `data_root` 修改为 ${OUT_DIR} 即可开始使用 DOTA 数据集训练。
