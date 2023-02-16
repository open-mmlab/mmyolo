# Preparing DOTA Dataset

## download dota dataset

The dota dataset can be downloaded from [here](https://captain-whu.github.io/DOTA/dataset.html).

The data structure is as follows:

```none
mmyolo
├── data
│   ├── DOTA
│   │   ├── train
│   │   ├── val
│   │   ├── test
```

## split dota dataset

Based on the configuration in the DOTA paper, we provide the config json file for image split.

`ss` means single scale split.
`ms` means multi scale split.

Please update the `img_dirs` and `ann_dirs` in json.

For single scale split:

```shell
mim run mmrotate img_split --base-json \
  tools/dataset_converters/dota_split/ss_trainval.json

mim run mmrotate img_split --base-json \
  tools/dataset_converters/dota_split/ss_test.json
```

For multi scale split:

```shell
mim run mmrotate img_split --base-json \
  tools/dataset_converters/dota_split/ms_trainval.json

mim run mmrotate img_split --base-json \
  tools/dataset_converters/dota_split/ms_test.json
```

The new data structure is as follows:

```none
mmyolo
├── data
│   ├── split_ss_dota
│   │   ├── trainval
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── annfiles
```

Then change `data_root` to `data/split_ss_dota` or `data/split_ms_dota`.
