# 基于 MMYOLO 的频高图实时目标检测 benchmark

## 数据集构建

数字频高图是获取电离层实时信息最重要的途径。电离层结构检测对精准提取电离层关键参数，具有非常重要的研究意义。

利用中国科学院在海南、武汉、怀来获取的不同季节的4311张频高图建立数据集，人工标注出E层、Es-c层、Es-l层、F1层、F2层、Spread F层共6种结构。[数据集下载](https://github.com/VoyagerXvoyagerx/Ionogram_detection/releases/download/Dataset/Iono4311.zip)

<div align=center>
<img width="40%" src="https://user-images.githubusercontent.com/67947949/223638535-c4583d88-aa5a-4f21-b35a-e6e8328c9bd4.jpg"/>

使用 labelme 标注的图像预览

</div>

1. 数据集格式转换

使用MMYOLO提供的 `tools/dataset_converters/labelme2coco.py` 脚本将 labelme 的 label 转换为 COCO label。

```shell
python tools/dataset_converters/labelme2coco.py --img-dir ./Iono4311/images \
                                                --labels-dir ./Iono4311/labels \
                                                --out ./Iono4311/annotations/annotations_all.json
```

2. 浏览数据集

使用下面的命令可以将 COCO 的 label 在图片上进行显示，这一步可以验证刚刚转换是否有问题。

```shell
python tools/analysis_tools/browse_coco_json.py --img-dir ./Iono4311/images \
                                                --ann-file ./Iono4311/annotations/annotations_all.json
```

3. 划分训练集、验证集、测试集

设置70%的图片为训练集，15%作为验证集，15%为测试集。

```shell
python tools/misc/coco_split.py --json ./Iono4311/annotations/annotations_all.json \
                                --out-dir ./Iono4311/annotations \
                                --ratios 0.7 0.15 0.15 \
                                --shuffle \
                                --seed 14
```

划分后的 File tree:

```shell
Iono4311/
├── annotations
│   ├── annotations_all.json
│   ├── class_with_id.txt
│   ├── test.json
│   ├── train.json
│   └── val.json
├── classes_with_id.txt
├── dataset_analysis.ipynb
├── dataset.ipynb
├── images
├── labels
├── test_images
├── train_images
└── val_images
```

使用以下代码可以统计数据集中各类别的实例数量：

```python
import json
import numpy as np
import pandas as pd


# 指定标注文件路径
ann_file_all = '/home/ubuntu/ionogram_detection/Iono4311/annotations/annotations_all.json'
ann_file_train = '/home/ubuntu/ionogram_detection/Iono4311/annotations/train.json'
ann_file_val = '/home/ubuntu/ionogram_detection/Iono4311/annotations/val.json'
ann_file_test = '/home/ubuntu/ionogram_detection/Iono4311/annotations/test.json'

for index, filename in enumerate((ann_file_all, ann_file_train, ann_file_val, ann_file_test)):
    with open(filename, 'r') as f:
        annotations = json.load(f)
    dataset = pd.DataFrame(np.zeros((1, 6), dtype=int), columns=['E', 'Esl', 'Esc', 'F1', 'F2', 'Fspread'])
    for ins in annotations["annotations"]:
        dataset.iloc[0, ins["category_id"]-1] += 1
    set_name = filename.split('/')
    print(set_name[-1][: -5], len(annotations["images"]), 'images')
    print(dataset, '\n')
    f.close()
```

得到如下输出，说明本数据集存在样本分布不均衡的现象。

```python
annotations_all 4311 images
      E  Esl  Esc    F1    F2  Fspread
0  2040  753  893  2059  4177      133

train 3019 images
      E  Esl  Esc    F1    F2  Fspread
0  1436  529  629  1459  2928       91

val 646 images
     E  Esl  Esc   F1   F2  Fspread
0  311  101  137  303  626       20

test 646 images
     E  Esl  Esc   F1   F2  Fspread
0  293  123  127  297  623       22
```

## 配置文件

配置文件在目录 `/configs/custom_dataset` 下。

1. 数据集可视化分析

```shell
python tools/analysis_tools/dataset_analysis.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-50e_ionogram.py \
                                                --out-dir output
```

<img width="100%" src="https://user-images.githubusercontent.com/67947949/223640412-4008a0a1-0626-419d-90bf-fb7ce6f26fc9.jpg"/>

E、Es-l、Esc、F1 类别以小目标居多，F2、Fspread 类主要是中等大小目标。

2. 可视化config中的数据处理部分

以 YOLOv5-s 为例：

```shell
python tools/analysis_tools/browse_dataset.py configs/custom_dataset/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram.py \
                                              --out-dir output --show-interval 1
```

根据配置文件中的 `train_pipeline`，训练时采用的数据增强策略包括：

- 马赛克增强
- 仿射变换
- Albumentations 数据增强工具包（包括多种数字图像处理方法）
- HSV随机增强图像
- 随机水平翻转

修改 `train_pipeline` 并可视化不同的数据增强方法：

| Aug Method | [config](/projects/misc/ionogram_detection/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram_aug0.py)            | [config](/projects/misc/ionogram_detection/yolov5/yolov5_s-v61_syncbn_fast_1xb32-100e_ionogram_mosaic.py)           | [config](/projects/misc/ionogram_detection/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram_mosaic_affine.py)   | [config](/projects/misc/ionogram_detection/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram.py)                  |
| ---------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| Mosaic     |                                                                                                                    | √                                                                                                                   | √                                                                                                                  | √                                                                                                                   |
| Affine     |                                                                                                                    |                                                                                                                     | √                                                                                                                  | √                                                                                                                   |
| Albu       |                                                                                                                    |                                                                                                                     |                                                                                                                    | √                                                                                                                   |
| HSV        |                                                                                                                    |                                                                                                                     |                                                                                                                    | √                                                                                                                   |
| Flip       |                                                                                                                    |                                                                                                                     |                                                                                                                    | √                                                                                                                   |
| 可视化     | <img src="https://user-images.githubusercontent.com/67947949/223640764-2f9798a5-90d7-4b69-9fd1-d0f54fae0433.png"/> | <img  src="https://user-images.githubusercontent.com/67947949/223640628-1ed7e911-8339-4c03-adcb-77eb3abe5a4c.png"/> | <img src="https://user-images.githubusercontent.com/67947949/223640839-df347b96-b649-4795-acf6-f1f42f53f9ea.png"/> | <img  src="https://user-images.githubusercontent.com/67947949/223640915-cd4dea0e-ac9f-43a8-a32c-fd81468ce187.png"/> |

3. 修改 Anchor 尺寸

使用分析工具中的 `tools/analysis_tools/optimize_anchors.py` 脚本得到适用于本数据集的先验锚框尺寸。

```shell
python tools/analysis_tools/optimize_anchors.py configs/custom_dataset/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram.py \
                                                --algorithm v5-k-means \
                                                --input-shape 640 640 \
                                                --prior-match-thr 4.0 \
                                                --out-dir work_dirs/dataset_analysis_5_s
```

4. 模型复杂度分析

根据配置文件，使用分析工具中的 `tools/analysis_tools/get_flops.py` 脚本可以得到模型的参数量、浮点计算量等信息。以 YOLOv5-s 为例：

```shell
python tools/analysis_tools/get_flops.py configs/custom_dataset/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram.py
```

得到如下输出，表示模型的浮点运算量为 7.947G，一共有 7.036M 个可学习参数。

```shell
==============================
Input shape: torch.Size([640, 640])
Model Flops: 7.947G
Model Parameters: 7.036M
==============================
```

## 训练和测试

1. 训练

```shell
python tools/train.py configs/custom_dataset/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram.py
```

调试技巧：在调试代码的过程中，有时需要训练几个 epoch，例如调试验证过程或者权重的保存是否符合期望。对于继承自 `BaseDataset` 的数据集（如本范例中的 `YOLOv5CocoDataset`），在 `train_dataloader` 中的 `dataset` 字段增加 `indices` 参数，即可指定每个 epoch 迭代的样本数，减少迭代时间。

```python
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=_base_.dataset_type,
            indices=200,  # 设置 indices=200，表示每个 epoch 只迭代 200 个样本
            data_root=data_root,
            metainfo=metainfo,
            ann_file=train_ann_file,
            data_prefix=dict(img=train_data_prefix),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=_base_.train_pipeline)))
```

2. 测试

```shell
python tools/test.py configs/custom_dataset/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram.py \
                     work_dirs/yolov5_s_100e/best_coco-test-0.584.pth
```

## 实验与结果分析

### 选择合适的 batch size

- Batch size 主导了训练速度。通常，理想的 batch size 是是硬件能支持的最大 batch size。
- 当显存占用没有达到饱和时，如果 batch size 翻倍，训练吞吐量也应该翻倍（或接近翻倍）。训练时间应该减半或接近减半。
- 使用**混合精度训练**可以加快训练速度、减小显存。在执行 `train.py` 脚本时添加 `--amp` 参数即可开启。

硬件信息：

- GPU：V100，显存 32G
- CPU：10核，内存 40G

实验结果：

| Model    | Epoch(best) | AMP   | Batchsize | Num workers | Memory Allocated | Training Time | Val mAP |
| -------- | ----------- | ----- | --------- | ----------- | ---------------- | ------------- | ------- |
| YOLOv5-s | 100(82)     | False | 32        | 6           | 35.07%           | 54 min        | 0.575   |
| YOLOv5-s | 100(96)     | True  | 32        | 6           | 24.93%           | 49 min        | 0.578   |
| YOLOv5-s | 100(100)    | False | 96        | 6           | 96.64%           | 48 min        | 0.571   |
| YOLOv5-s | 100(100)    | True  | 96        | 6           | 54.66%           | **37** min    | 0.575   |
| YOLOv5-s | 100(90)     | True  | 144       | 6           | 77.06%           | 39 min        | 0.573   |
| YOLOv5-s | 200(148)    | True  | 96        | 6           | 54.66%           | 72 min        | 0.575   |
| YOLOv5-s | 200(188)    | True  | 96        | **8**       | 54.66%           | 67 min        | 0.576   |

不同 batch size 的训练过程中，数据加载时间 `data_time` 占每步总时长的比例

<div align=center>
<img width="60%" src="https://user-images.githubusercontent.com/67947949/223635966-948c8424-1ba8-4df0-92d7-079029dc1231.png">
</div>

分析结果，可以得出以下结论：

- 使用混合精度训练队模型表现影响很小（约百分之零点几），并且可以明显减少显存占用。
- Batch size 增加三倍，和训练时长并没有相应地减小3倍。根据训练记录，batch size 越大，`data_time` 也越大，说明数据加载成为了限制训练速度的瓶颈。增大加载数据的进程数（`num_workers`）可以加快数据加载。

### 消融实验

为了得到使用于本数据集的训练流水线，以 YOLOv5-s 模型为例，进行以下的消融实验。

#### 不同数据增强方法

| Aug Method | [config](/projects/misc/ionogram_detection/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram_aug0.py) | [config](/projects/misc/ionogram_detection/yolov5/yolov5_s-v61_syncbn_fast_1xb32-100e_ionogram_mosaic.py) | [config](/projects/misc/ionogram_detection/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram_mosaic_affine.py) | [config](/projects/misc/ionogram_detection/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram_mosaic_affine_albu_hsv.py) | [config](/projects/misc/ionogram_detection/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram.py) |
| ---------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Mosaic     |                                                                                                         | √                                                                                                         | √                                                                                                                | √                                                                                                                         | √                                                                                                  |
| Affine     |                                                                                                         |                                                                                                           | √                                                                                                                | √                                                                                                                         | √                                                                                                  |
| Albu       |                                                                                                         |                                                                                                           |                                                                                                                  | √                                                                                                                         | √                                                                                                  |
| HSV        |                                                                                                         |                                                                                                           |                                                                                                                  | √                                                                                                                         | √                                                                                                  |
| Flip       |                                                                                                         |                                                                                                           |                                                                                                                  |                                                                                                                           | √                                                                                                  |
| Val mAP    | 0.507                                                                                                   | 0.550                                                                                                     | 0.572                                                                                                            | 0.567                                                                                                                     | 0.575                                                                                              |

结果表明，马赛克增强和随机仿射变换可以对验证集表现带来明显的提升。

#### 是否使用预训练权重

在配置文件中，修改 `load_from = None` 即可不使用预训练权重。对不使用预训练权重的实验，将基础学习率增大四倍，训练轮数增加至 200 轮，保证模型得到充分的训练。

| Model    | Epoch(best) | FLOPs(G) | Params(M) | Pretrain | Val mAP | Config                                                                                                  |
| -------- | ----------- | -------- | --------- | -------- | ------- | ------------------------------------------------------------------------------------------------------- |
| YOLOv5-s | 100(82)     | 7.95     | 7.04      | Coco     | 0.575   | [config](/projects/misc/ionogram_detection/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram.py)      |
| YOLOv5-s | 200(145)    | 7.95     | 7.04      | None     | 0.565   | [config](/projects/misc/ionogram_detection/yolov5/yolov5_s-v61_syncbn_fast_1xb96-200e_ionogram_pre0.py) |
| YOLOv6-s | 100(54)     | 24.2     | 18.84     | Coco     | 0.584   | [config](/projects/misc/ionogram_detection/yolov6/yolov6_s_syncbn_fast_1xb32-100e_ionogram.py)          |
| YOLOv6-s | 200(188)    | 24.2     | 18.84     | None     | 0.557   | [config](/projects/misc/ionogram_detection/yolov6/yolov6_s_syncbn_fast_1xb32-200e_ionogram_pre0.py)     |

<div align=center>
<img width="60%" src="https://user-images.githubusercontent.com/67947949/223641016-9ded0d11-62b8-45f4-be5b-bd4ffae3ec21.png">

训练过程中的损失下降对比图

</div>

损失下降的过程表明，使用预训练权重时，loss 下降得更快。可见即使是自然图像数据集上预训练的模型，在雷达图像数据集上微调，也可以加快收敛。

### 频高图检测 benchmark

| Model       | epoch(best) | FLOPs(G) | Params(M) | pretrain | val mAP | test mAP | Config                                                                                             | Log                                                                                                           |
| ----------- | ----------- | -------- | --------- | -------- | ------- | -------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| YOLOv5-s    | 100(82)     | 7.95     | 7.04      | Coco     | 0.575   | 0.584    | [config](/projects/misc/ionogram_detection/yolov5/yolov5_s-v61_syncbn_fast_1xb96-100e_ionogram.py) | [log](https://github.com/VoyagerXvoyagerx/Ionogram_detection/blob/main/logs/yolov5_s_20230105_213510.json)    |
| YOLOv5-m    | 100(70)     | 24.05    | 20.89     | Coco     | 0.587   | 0.586    | [config](/projects/misc/ionogram_detection/yolov5/yolov5_m-v61_syncbn_fast_1xb32-100e_ionogram.py) | [log](https://github.com/VoyagerXvoyagerx/Ionogram_detection/blob/main/logs/yolov5_m_20230106_004642.json)    |
| YOLOv6-s    | 100(54)     | 24.2     | 18.84     | Coco     | 0.584   | 0.594    | [config](/projects/misc/ionogram_detection/yolov6/yolov6_s_syncbn_fast_1xb32-100e_ionogram.py)     | [log](https://github.com/VoyagerXvoyagerx/Ionogram_detection/blob/main/logs/yolov6_s_20230107_003207.json)    |
| YOLOv6-m    | 100(76)     | 37.08    | 44.42     | Coco     | 0.590   | 0.590    | [config](/projects/misc/ionogram_detection/yolov6/yolov6_m_syncbn_fast_1xb32-100e_ionogram.py)     | [log](https://github.com/VoyagerXvoyagerx/Ionogram_detection/blob/main/logs/yolov6_m_20230107_201029.json)    |
| YOLOv6-l    | 100(76)     | 71.33    | 58.47     | Coco     | 0.605   | 0.597    | [config](/projects/misc/ionogram_detection/yolov6/yolov6_l_syncbn_fast_1xb32-100e_ionogram.py)     | [log](https://github.com/VoyagerXvoyagerx/Ionogram_detection/blob/main/logs/yolov6_l_20230108_005634.json)    |
| YOLOv7-tiny | 100(78)     | 6.57     | 6.02      | Coco     | 0.549   | 0.568    | [config](/projects/misc/ionogram_detection/yolov7/yolov7_tiny_syncbn_fast_1xb16-100e_ionogram.py)  | [log](https://github.com/VoyagerXvoyagerx/Ionogram_detection/blob/main/logs/yolov7_tiny_20230215_202837.json) |
| YOLOv7-x    | 100(58)     | 94.27    | 70.85     | Coco     | 0.602   | 0.595    | [config](/projects/misc/ionogram_detection/yolov7/yolov7_x_syncbn_fast_1xb16-100e_ionogram.py)     | [log](https://github.com/VoyagerXvoyagerx/Ionogram_detection/blob/main/logs/yolov7_x_20230110_165832.json)    |
| rtmdet-s    | 100(64)     | 14.76    | 8.86      | Coco     | 0.581   | 0.571    | [config](/projects/misc/ionogram_detection/rtmdet/rtmdet_s_syncbn_fast_1xb8-100e_ionogram.py)      | [log](https://github.com/VoyagerXvoyagerx/Ionogram_detection/blob/main/logs/rtmdet_s_20230215_211817.json)    |
