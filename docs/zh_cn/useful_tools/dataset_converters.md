# 数据集转换

文件夹 `tools/data_converters/` 目前包含 `ballon2coco.py`、`yolo2coco.py` 和 `labelme2coco.py` 三个数据集转换工具。

- `ballon2coco.py` 将 `balloon` 数据集（该小型数据集仅作为入门使用）转换成 COCO 的格式。

```shell
python tools/dataset_converters/balloon2coco.py
```

- `yolo2coco.py` 将 `yolo-style` **.txt** 格式的数据集转换成 COCO 的格式，请按如下方式使用：

```shell
python tools/dataset_converters/yolo2coco.py /path/to/the/root/dir/of/your_dataset
```

使用说明：

1. `image_dir` 是需要你传入的待转换的 yolo 格式数据集的根目录，内应包含 `images` 、 `labels` 和 `classes.txt` 文件， `classes.txt` 是当前 dataset 对应的类的声明，一行一个类别。
   `image_dir` 结构如下例所示：

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

2. 脚本会检测 `image_dir` 下是否已有的 `train.txt` 、 `val.txt` 和 `test.txt` 。若检测到文件，则会按照类别进行整理， 否则默认不需要分类。故请确保对应的 `train.txt` 、 `val.txt` 和 `test.txt` 要在 `image_dir` 内。文件内的图片路径必须是**绝对路径**。
3. 脚本会默认在 `image_dir` 目录下创建 `annotations` 文件夹并将转换结果存在这里。如果在 `image_dir` 下没找到分类文件，输出文件即为一个 `result.json`，反之则会生成需要的 `train.json` 、 `val.json`、 `test.json`，脚本完成后 `annotations` 结构可如下例所示：

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
