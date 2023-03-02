# 可视化数据集分析结果

脚本 `tools/analysis_tools/dataset_analysis.py` 能够帮助用户得到四种功能的结果图，并将图片保存到当前运行目录下的 `dataset_analysis` 文件夹中。

关于该脚本的功能的说明：

通过 `main()` 的数据准备，得到每个子函数所需要的数据。

功能一：显示类别和 bbox 实例个数的分布图，通过子函数 `show_bbox_num` 生成。

<img src="https://user-images.githubusercontent.com/90811472/200314770-4fb21626-72f2-4a4c-be5d-bf860ad830ec.jpg"/>

功能二：显示类别和 bbox 实例宽、高的分布图，通过子函数 `show_bbox_wh` 生成。

<img src="https://user-images.githubusercontent.com/90811472/200315007-96e8e795-992a-4c72-90fa-f6bc00b3f2c7.jpg"/>

功能三：显示类别和 bbox 实例宽/高比例的分布图，通过子函数 `show_bbox_wh_ratio` 生成。

<img src="https://user-images.githubusercontent.com/90811472/200315044-4bdedcf6-087a-418e-8fe8-c2d3240ceba8.jpg"/>

功能四：基于面积规则下，显示类别和 bbox 实例面积的分布图，通过子函数 `show_bbox_area` 生成。

<img src="https://user-images.githubusercontent.com/90811472/200315075-71680fe2-db6f-4981-963e-a035c1281fc1.jpg"/>

打印列表显示，通过脚本中子函数 `show_class_list` 和 `show_data_list` 生成。

<img src="https://user-images.githubusercontent.com/90811472/200315152-9d6df91c-f2d2-4bba-9f95-b790fac37b62.jpg"/>

```shell
python tools/analysis_tools/dataset_analysis.py ${CONFIG} \
                                                [-h] \
                                                [--val-dataset ${TYPE}] \
                                                [--class-name ${CLASS_NAME}] \
                                                [--area-rule ${AREA_RULE}] \
                                                [--func ${FUNC}] \
                                                [--out-dir ${OUT_DIR}]
```

例子：

1. 使用 `config` 文件 `configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py` 分析数据集，其中默认设置:数据加载类型为 `train_dataset` ，面积规则设置为 `[0,32,96,1e5]` ,生成包含所有类的结果图并将图片保存到当前运行目录下 `./dataset_analysis` 文件夹中：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py
```

2. 使用 `config` 文件 `configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py` 分析数据集，通过 `--val-dataset` 设置将数据加载类型由默认的 `train_dataset` 改为 `val_dataset`：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py \
                                               --val-dataset
```

3. 使用 `config` 文件 `configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py` 分析数据集，通过 `--class-name` 设置将生成所有类改为特定类显示，以显示 `person` 为例：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py \
                                               --class-name person
```

4. 使用 `config` 文件 `configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py` 分析数据集，通过 `--area-rule` 重新定义面积规则，以 `30 70 125` 为例,面积规则变为 `[0,30,70,125,1e5]`：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py \
                                               --area-rule 30 70 125
```

5. 使用 `config` 文件 `configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py` 分析数据集，通过 `--func` 设置，将显示四个功能效果图改为只显示 `功能一` 为例：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py \
                                               --func show_bbox_num
```

6. 使用 `config` 文件 `configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py` 分析数据集，通过 `--out-dir` 设置修改图片保存地址，以 `work_dirs/dataset_analysis` 地址为例：

```shell
python tools/analysis_tools/dataset_analysis.py configs/yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py \
                                               --out-dir work_dirs/dataset_analysis
```
