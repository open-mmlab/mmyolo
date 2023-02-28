# 可视化数据集

```shell
python tools/analysis_tools/browse_dataset.py \
    ${CONFIG_FILE} \
    [-o, --out-dir ${OUTPUT_DIR}] \
    [-p, --phase ${DATASET_PHASE}] \
    [-n, --show-number ${NUMBER_IMAGES_DISPLAY}] \
    [-i, --show-interval ${SHOW_INTERRVAL}] \
    [-m, --mode ${DISPLAY_MODE}] \
    [--cfg-options ${CFG_OPTIONS}]
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- `-o, --out-dir`: 保存图片文件夹，如果没有指定，默认为 `'./output'`。
- **`-p, --phase`**: 可视化数据集的阶段，只能为 `['train', 'val', 'test']` 之一，默认为 `'train'`。
- **`-n, --show-number`**: 可视化样本数量。如果没有指定，默认展示数据集的所有图片。
- **`-m, --mode`**: 可视化的模式，只能为 `['original', 'transformed', 'pipeline']` 之一。 默认为 `'transformed'`。
- `--cfg-options` : 对配置文件的修改，参考[学习配置文件](../tutorials/config.md)。

```shell
`-m, --mode` 用于设置可视化的模式，默认设置为 'transformed'。
- 如果 `--mode` 设置为 'original'，则获取原始图片；
- 如果 `--mode` 设置为 'transformed'，则获取预处理后的图片；
- 如果 `--mode` 设置为 'pipeline'，则获得数据流水线所有中间过程图片。
```

**示例**：

1. **'original'** 模式 ：

```shell
python ./tools/analysis_tools/browse_dataset.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py --phase val --out-dir tmp --mode original
```

- `--phase val`: 可视化验证集, 可简化为 `-p val`;
- `--out-dir tmp`: 可视化结果保存在 "tmp" 文件夹, 可简化为 `-o tmp`;
- `--mode original`: 可视化原图, 可简化为 `-m original`;
- `--show-number 100`: 可视化100张图，可简化为 `-n 100`;

2. **'transformed'** 模式 ：

```shell
python ./tools/analysis_tools/browse_dataset.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py
```

3. **'pipeline'** 模式 ：

```shell
python ./tools/analysis_tools/browse_dataset.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py -m pipeline
```

<div align=center>
<img src="https://user-images.githubusercontent.com/45811724/204810831-0fbc7f1c-0951-4be1-a11c-491cf0d194f6.png" alt="Image">
</div>
