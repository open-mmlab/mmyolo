# Dataset Conversion

The folder `tools/data_converters` currently contains `ballon2coco.py`, `yolo2coco.py`, and `labelme2coco.py` - three dataset conversion tools.

- `ballon2coco.py` converts the `balloon` dataset (this small dataset is for starters only) to COCO format.

```shell
python tools/dataset_converters/balloon2coco.py
```

- `yolo2coco.py` converts a dataset from `yolo-style` **.txt** format to COCO format, please use it as follows:

```shell
python tools/dataset_converters/yolo2coco.py /path/to/the/root/dir/of/your_dataset
```

Instructions:

1. `image_dir` is the root directory of the yolo-style dataset you need to pass to the script, which should contain `images`, `labels`, and `classes.txt`. `classes.txt` is the class declaration corresponding to the current dataset. One class a line. The structure of the root directory should be formatted as this example shows:

```bash
.
└── $ROOT_PATH
    ├── classes.txt
    ├── labels
    │    ├── a.txt
    │    ├── b.txt
    │    └── ...
    ├── images
    │    ├── a.jpg
    │    ├── b.png
    │    └── ...
    └── ...
```

2. The script will automatically check if `train.txt`, `val.txt`, and `test.txt` have already existed under `image_dir`. If these files are located, the script will organize the dataset accordingly. Otherwise, the script will convert the dataset into one file. The image paths in these files must be **ABSOLUTE** paths.
3. By default, the script will create a folder called `annotations` in the `image_dir` directory which stores the converted JSON file. If `train.txt`, `val.txt`, and `test.txt` are not found, the output file is `result.json`. Otherwise, the corresponding JSON file will be generated, named as `train.json`, `val.json`, and `test.json`. The `annotations` folder may look similar to this:

```bash
.
└── $ROOT_PATH
    ├── annotations
    │    ├── result.json
    │    └── ...
    ├── classes.txt
    ├── labels
    │    ├── a.txt
    │    ├── b.txt
    │    └── ...
    ├── images
    │    ├── a.jpg
    │    ├── b.png
    │    └── ...
    └── ...
```
