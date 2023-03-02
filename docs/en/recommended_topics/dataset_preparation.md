# Dataset preparation and description

## DOTA Dataset

### Download dataset

The DOTA dataset can be downloaded from [DOTA](https://captain-whu.github.io/DOTA/dataset.html)
or [OpenDataLab](https://opendatalab.org.cn/DOTA_V1.0).

We recommend using [OpenDataLab](https://opendatalab.org.cn/DOTA_V1.0) to download the dataset, as the folder structure has already been arranged as needed and can be directly extracted without the need to adjust the folder structure.

Please unzip the file and place it in the following structure.

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

The folder ending with reclabelTxt stores the labels for the horizontal boxes and is not used when slicing.

### Split DOTA dataset

Script `tools/dataset_converters/dota/dota_split.py` can split and prepare DOTA dataset.

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

shapely is required, please install shapely first by `pip install shapely`.

**Description of all parameters**：

- `--split-config` : The split config for image slicing.
- `--data-root`: Root dir of DOTA dataset.
- `--out-dir`: Output dir for split result.
- `--ann-subdir`: The subdir name for annotation. Defaults to `labelTxt-v1.0`.
- `--phase`: Phase of the data set to be prepared. Defaults to `trainval test`
- `--nproc`: Number of processes. Defaults to 8.
- `--save-ext`: Extension of the saved image. Defaults to `png`
- `--overwrite`: Whether to allow overwrite if annotation folder exist.

Based on the configuration in the DOTA paper, we provide two commonly used split config.

- `./split_config/single_scale.json` means single-scale split.
- `./split_config/multi_scale.json` means multi-scale split.

DOTA dataset usually uses the trainval set for training and the test set for online evaluation, since most papers
provide the results of online evaluation. If you want to evaluate the model performance locally firstly, please split
the train set and val set.

Examples:

Split DOTA trainval set and test set with single scale.

```shell
python tools/dataset_converters/dota/dota_split.py
    --split-config 'tools/dataset_converters/dota/split_config/single_scale.json'
    --data-root ${DATA_ROOT} \
    --out-dir ${OUT_DIR}
```

If you want to split DOTA-v1.5 dataset, which have different annotation dir 'labelTxt-v1.5'.

```shell
python tools/dataset_converters/dota/dota_split.py
    --split-config 'tools/dataset_converters/dota/split_config/single_scale.json'
    --data-root ${DATA_ROOT} \
    --out-dir ${OUT_DIR} \
    --ann-subdir 'labelTxt-v1.5'
```

If you want to split DOTA train and val set with single scale.

```shell
python tools/dataset_converters/dota/dota_split.py
    --split-config 'tools/dataset_converters/dota/split_config/single_scale.json'
    --data-root ${DATA_ROOT} \
    --phase train val \
    --out-dir ${OUT_DIR}
```

For multi scale split:

```shell
python tools/dataset_converters/dota/dota_split.py
    --split-config 'tools/dataset_converters/dota/split_config/multi_scale.json'
    --data-root ${DATA_ROOT} \
    --out-dir ${OUT_DIR}
```

The new data structure is as follows:

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

Then change `data_root` to ${OUT_DIR}.
