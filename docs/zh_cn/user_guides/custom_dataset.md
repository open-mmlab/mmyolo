# 自定义数据集 标注+训练+测试+部署 全流程

本章节会介绍从 用户自定义图片数据集标注 到 最终进行训练和部署 的整体流程。流程步骤概览如下：

01. 数据集准备：`tools/misc/download_dataset.py`
02. 使用 [labelme](https://github.com/wkentaro/labelme) 进行数据集标注：`demo/image_demo.py` + labelme
03. 使用脚本转换成 COCO 数据集格式：`tools/dataset_converters/labelme2coco.py`
04. 数据集划分：`tools/misc/coco_split.py`
05. 根据数据集内容新建 config 文件
06. 数据集分析：`tools/analysis_tools/dataset_analysis.py`
07. 优化 Anchor 尺寸：`tools/analysis_tools/optimize_anchors.py`
08. 训练：`tools/train.py`
09. 推理：`demo/image_demo.py`
10. 部署

下面详细介绍每一步。

## 1. 数据集准备

- 如果自己没有数据集，可以使用本教程提供的一个 `cat` 数据集（本 `cat` 数据集由 @RangeKing 提供原始图片，由 @PeterH0323 进行数据清洗）

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205423220-c4b8f2fd-22ba-4937-8e47-1b3f6a8facd8.png" alt="cat dataset"/>
</div>

下载命令：

```shell
python tools/misc/download_dataset.py --dataset-name cat --save-dir ./data/cat --unzip --delete
```

该命令会自动下载数据集到 `./data/cat` 文件夹中，该文件的目录结构是：

```shell
.
└── ./data/cat
    ├── images # 图片文件
    │    ├── image1.jpg
    │    ├── image2.png
    │    └── ...
    ├── labels # labelme 标注文件
    │    ├── image1.json
    │    ├── image2.json
    │    └── ...
    ├── annotations # 数据集划分的 COCO 文件
    │    ├── annotations_all.json # 全量数据的 COCO label 文件
    │    ├── trainval.json # 划分比例 80% 的数据
    │    └── test.json # 划分比例 20% 的数据
    └── class_with_id.txt # id + class_name 文件
```

**Tips**：这个数据集可以直接训练，如果您想体验整个流程的话，可以将 `images` 文件夹**以外的**其余文件都删除。

- 如你已经有数据，可以将其组成下面的结构

```shell
.
└── $DATA_ROOT
    └── images
         ├── image1.jpg
         ├── image2.png
         └── ...
```

## 2. 使用 labelme 进行数据集标注

通常，标注有 2 种方法：

- 软件或者算法辅助 + 人工修正 label
- 仅人工标注

## 2.1 软件或者算法辅助 + 人工修正 label

辅助标注的原理是用已有模型进行推理，将得出的推理信息保存为标注软件 label 文件格式。

**Tips**：如果已有模型典型的如 COCO 预训练模型没有你自定义新数据集的类别，建议先人工打 100 张左右的图片 label，训练个初始模型，然后再进行辅助标注。

人工操作标注软件加载生成好的 label 文件，只需要检查每张图片的目标是否标准，以及是否有漏掉的目标。

【辅助 + 人工标注】这种方式可以节省很多时间和精力，达到降本提速的目的。

下面会分别介绍其过程：

### 2.1.1 软件或者算法辅助

MMYOLO 提供的模型推理脚本 `demo/image_demo.py` 设置 `--to-labelme` 可以生成 labelme 格式 label 文件，具体用法如下：

```shell
python demo/image_demo.py img \
                          config \
                          checkpoint
                          [--out-dir OUT_DIR] \
                          [--device DEVICE] \
                          [--show] \
                          [--deploy] \
                          [--score-thr SCORE_THR] \
                          [--class-name CLASS_NAME]
                          [--to-labelme]
```

其中：

- `img`： 图片的路径，支持文件夹、文件、URL；
- `config`：用到的模型 config 文件路径；
- `checkpoint`：用到的模型权重文件路径；
- `--out-dir`：推理结果输出到指定目录下，默认为 `./output`，当 `--show` 参数存在时，不保存检测结果；
- `--device`：使用的计算资源，包括 `CUDA`, `CPU` 等，默认为 `cuda:0`；
- `--show`：使用该参数表示在屏幕上显示检测结果，默认为 `False`；
- `--deploy`：是否切换成 deploy 模式；
- `--score-thr`：置信度阈值，默认为 `0.3`；
- `--to-labelme`：是否导出 `labelme` 格式的 label 文件，不可以与 `--show` 参数同时存在

例子：

这里使用 YOLOv5-s 作为例子来进行辅助标注刚刚下载的 `cat` 数据集，先下载 YOLOv5-s 的权重:

```shell
mkdir work_dirs
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth -P ./work_dirs
```

由于 COCO 80 类数据集中已经包括了 `cat` 这一类，因此我们可以直接加载 COCO 预训练权重进行辅助标注。

```shell
python demo/image_demo.py ./data/cat/images \
                          ./configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                          ./work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
                          --out-dir ./data/cat/labels \
                          --class-name cat \
                          --to-labelme
```

**Tips**：

- 如果你的数据集需要标注多类，可以采用类似 `--class-name class1 class2` 格式输入；
- 如果全部输出，则删掉 `--class-name` 这个 flag 即可全部类都输出。

生成的 label 文件会在 `--out-dir` 中:

```shell
.
└── $OUT_DIR
    ├── image1.json
    ├── image1.json
    └── ...
```

### 2.1.2 人工标注

本教程使用的标注软件是 [labelme](https://github.com/wkentaro/labelme)

- 安装 labelme

```shell
pip install labelme
```

- 启动 labelme

```shell
labelme ${图片文件夹路径（即上一步的图片文件夹）} \
        --output ${label文件所处的文件夹路径（即上一步的 --out-dir）} \
        --autosave \
        --nodata
```

其中：

- `--output`：labelme 标注文件保存路径，如果该路径下已经存在部分图片的标注文件，则会进行加载；
- `--autosave`：标注文件自动保存，会略去一些繁琐的保存步骤；
- `--nodata`：每张图片的标注文件中不保存图片的 base64 编码，设置了这个 flag 会大大减少标注文件的大小。

例子：

```shell
labelme ./data/cat/images --output ./data/cat/labels --autosave --nodata
```

输入命令之后 labelme 就会启动，然后进行 label 检查即可。如果 labelme 启动失败，命令行输入 `export QT_DEBUG_PLUGINS=1` 查看具体缺少什么库，安装一下即可。

**注意：标注的时候务必使用 `rectangle`，快捷键 `Ctrl + R`（如下图）**

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/204076212-86dab4fa-13dd-42cd-93d8-46b04b864449.png" alt="rectangle"/>
</div>

## 2.2 仅人工标注

步骤和 【1.1.2 人工标注】 相同，只是这里是直接标注，没有预先生成的 label 。

## 3. 使用脚本转换成 COCO 数据集格式

### 3.1 使用脚本转换

MMYOLO 提供脚本将 labelme 的 label 转换为 COCO label

```shell
python tools/dataset_converters/labelme2coco.py --img-dir ${图片文件夹路径} \
                                                --labels-dir ${label 文件夹位置} \
                                                --out ${输出 COCO label json 路径} \
                                                [--class-id-txt ${class_with_id.txt 路径}]
```

其中：
`--class-id-txt`：是数据集 `id class_name` 的 `.txt` 文件：

- 如果不指定，则脚本会自动生成，生成在 `--out` 同级的目录中，保存文件名为 `class_with_id.txt`；
- 如果指定，脚本仅会进行读取但不会新增或者覆盖，同时，脚本里面还会判断是否存在 `.txt` 中其他的类，如果出现了会报错提示，届时，请用户检查 `.txt` 文件并加入新的类及其 `id`。

`.txt` 文件的例子如下（ `id` 可以和 COCO 一样，从 `1` 开始）：

```text
1 cat
2 dog
3 bicycle
4 motorcycle

```

例子：

```shell
python tools/dataset_converters/labelme2coco.py --img-dir ./data/cat/image \
                                                --labels-dir ./data/cat/labels \
                                                --out ./data/cat/annotations/annotations_all.json
```

### 3.2 检查转换的 COCO label

使用下面的命令可以将 COCO 的 label 在图片上进行显示，这一步可以验证刚刚转换是否有问题：

```shell
python tools/analysis_tools/browse_coco_json.py --img-dir ${图片文件夹路径} \
                                                --ann-file ${COCO label json 路径}
```

例子：

```shell
python tools/dataset_converters/labelme2coco.py --img-dir ./data/cat/image \
                                                --ann-file ./data/cat/annotations/annotations_all.json
```

关于 `tools/analysis_tools/browse_coco_json.py` 的更多用法请参考 [可视化 COCO label](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#coco)。

## 4. 数据集划分

```shell
python tools/misc/coco_split.py --json ${COCO label json 路径} \
                                --out-dir ${划分 label json 保存根路径} \
                                --ratios ${划分比例} \
                                [--shuffle] \
                                [--seed ${划分的随机种子}]
```

其中：

- `--ratios`：划分的比例，如果只设置了 2 个，则划分为 `trainval + test`，如果设置为 3 个，则划分为 `train + val + test`。支持两种格式 —— 整数、小数：
  - 整数：按比分进行划分，代码中会进行归一化之后划分数据集。例子： `--ratio 2 1 1`（代码里面会转换成 `0.5 0.25 0.25`） or `--ratio 3 1`（代码里面会转换成 `0.75 0.25`）
  - 小数：划分为比例。**如果加起来不为 1 ，则脚本会进行自动归一化修正**。例子： `--ratio 0.8 0.1 0.1` or `--ratio 0.8 0.2`
- `--shuffle`: 是否打乱数据集再进行划分；
- `--seed`：设定划分的随机种子，不设置的话自动生成随机种子。

例子：

```shell
python tools/misc/coco_split.py --json ./data/cat/annotations/annotations_all.json \
                                --out-dir ./data/cat/annotations \
                                --ratios 0.8 0.2 \
                                --shuffle \
                                --seed 666
```

## 5. 根据数据集内容新建 config 文件

确保数据集目录是这样的：

```shell
.
└── $DATA_ROOT
    ├── annotations
    │    ├── train.json # or trainval.json
    │    ├── val.json # optional
    │    └── test.json
    ├── images
    │    ├── image1.jpg
    │    ├── image1.png
    │    └── ...
    └── ...
```

因为是我们自定义的数据集，所以我们需要自己新建一个 config 并加入需要修改的部分信息。

关于新的 config 的命名：

- 这个 config 继承的是 `yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py`；
- 训练的类以本教程提供的数据集中的类 `cat` 为例（如果是自己的数据集，可以自定义类型的总称）；
- 本教程测试的显卡型号是 1 x 3080Ti 12G 显存，电脑内存 32G，可以训练 YOLOv5-s 最大批次是 `batch size = 32`（详细机器资料可见附录）；
- 训练轮次是 `100 epoch`。

综上所述：可以将其命名为 `yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py`。

我们可以在 configs 目录下新建一个新的目录 `custom_dataset`，同时在里面新建该 config 文件，并添加以下内容：

```python
_base_ = '../yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

max_epochs = 100  # 训练的最大 epoch
data_root = './data/cat/'  # 数据集目录的绝对路径

# 结果保存的路径，可以省略，省略保存的文件名位于 work_dirs 下 config 同名的文件夹中
# 如果某个 config 只是修改了部分参数，修改这个变量就可以将新的训练文件保存到其他地方
work_dir = './work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat'

# load_from 可以指定本地路径或者 URL，设置了 URL 会自动进行下载，因为上面已经下载过，我们这里设置本地路径
# 因为我们是微调数据集，故这里需要 load MMYOLO 已经训练好的预训练模型，这样可以加快收敛速度的同时保证精度
load_from = './work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

# 根据自己的 GPU 情况，修改 batch size，YOLOv5-s 默认为 8卡 x 16bs
train_batch_size_per_gpu = 32
train_num_workers = 4  # 推荐使用 train_num_workers = nGPU x 4

save_epoch_intervals = 2  # 每 interval 轮迭代进行一次保存一次权重

# 根据自己的 GPU 情况，修改 base_lr，修改的比例是 base_lr_default * (your_bs / default_bs)
base_lr = _base_.base_lr / 4

anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]

class_name = ('cat', )  # 根据 class_with_id.txt 类别信息，设置 class_name
num_classes = len(class_name)
metainfo = dict(
    CLASSES=class_name,
    PALETTE=[(220, 20, 60)]  # 画图时候的颜色，随便设置即可
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=10,  # 第几个epoch后验证，这里设置 10 是因为前 10 个 epoch 精度不高，测试意义不大，故跳过
    val_interval=save_epoch_intervals  # 每 val_interval 轮迭代进行一次测试评估
)

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),
        loss_cls=dict(loss_weight=0.5 *
                      (num_classes / 80 * 3 / _base_.num_det_layers))))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,  # 数据量太少的话，可以使用 RepeatDataset 来增量数据，这里设置 5 是 5 倍
        dataset=dict(
            type=_base_.dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/trainval.json',
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=_base_.train_pipeline)))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/trainval.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/trainval.json')
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=base_lr))

default_hooks = dict(
    # 设置间隔多少个 epoch 保存模型，以及保存模型最多几个，`save_best` 是另外保存最佳模型（推荐）
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=5,
        save_best='auto'),
    # logger 输出的间隔
    logger=dict(type='LoggerHook', interval=10))
```

**Tips**：我们在 `projects/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py` 放了一份相同的 config 文件，用户可以选择复制到 `configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py` 路径直接开始训练。

## 6. 数据集分析

脚本 `tools/analysis_tools/dataset_analysis.py` 能够帮助用户得到数据集的分析图，

可以生成四种分析图：

- 显示类别和 bbox 实例个数的分布图：`show_bbox_num`
- 显示类别和 bbox 实例宽、高的分布图：`show_bbox_wh`
- 显示类别和 bbox 实例宽/高比例的分布图：`show_bbox_wh_ratio`
- 基于面积规则下，显示类别和 bbox 实例面积的分布图：`show_bbox_area`

脚本使用方式如下：

```shell
python tools/analysis_tools/dataset_analysis.py ${CONFIG} \
                                                [--val-dataset ${TYPE}] \
                                                [--class-name ${CLASS_NAME}] \
                                                [--area-rule ${AREA_RULE}] \
                                                [--func ${FUNC}] \
                                                [--out-dir ${OUT_DIR}]
```

例子：

以本教程的数据集 cat config 为例：

查看训练集情况

```shell
python tools/analysis_tools/dataset_analysis.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                                                --output-dir work_dirs/dataset_analysis_cat/train_dataset
```

查看验证集情况

```shell
python tools/analysis_tools/dataset_analysis.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                                                --output-dir work_dirs/dataset_analysis_cat/val_dataset \
                                                --val-dataset
```

效果：

<div align="center">
  <img alt="YOLOv5CocoDataset_bbox_area" src="https://user-images.githubusercontent.com/25873202/205420210-e31c8643-1a12-4b79-b423-b04adcc68deb.jpg" width="45%">
  <img alt="YOLOv5CocoDataset_bbox_wh" src="https://user-images.githubusercontent.com/25873202/205420277-ccd58eb5-1e36-425c-a221-934e54175ae7.jpg" width="45%">
</div>
<div align="center">
  <img alt="YOLOv5CocoDataset_bbox_num" src="https://user-images.githubusercontent.com/25873202/205420241-f26c2310-8d4b-4b53-8331-bc2a67d62ce7.jpg" width="45%">
  <img alt="YOLOv5CocoDataset_bbox_ratio" src="https://user-images.githubusercontent.com/25873202/205420256-8151f01a-2f54-46df-8a9f-2c5de05bbbc8.jpg" width="45%">
</div>

关于 `tools/analysis_tools/dataset_analysis.py` 的更多用法请参考 [可视化数据集分析](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#id4)。

## 7. 优化 Anchor 尺寸

脚本 `tools/analysis_tools/optimize_anchors.py` 支持 YOLO 系列中三种锚框生成方式，分别是 `k-means`、`differential_evolution`、`v5-k-means`.

本示例使用的是 YOLOv5 进行训练，使用的是 `640 x 640` 的输入大小，故下面介绍 `v5-k-means` 的命令：

```shell
python tools/analysis_tools/optimize_anchors.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                                                --algorithm v5-k-means \
                                                --input-shape 640 640 \
                                                --prior-match-thr 4.0 \
                                                --out-dir work_dirs/dataset_analysis_cat
```

经过计算的 Anchor 如下：

<img alt="Anchor" src="https://user-images.githubusercontent.com/25873202/205422434-1a68cded-b055-42e9-b01c-3e51f8f5ef81.png">

修改 config 文件里面的 `anchors` 变量：

```python
anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]
```

关于 `tools/analysis_tools/optimize_anchors.py` 的更多用法请参考 [优化锚框尺寸](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#id8)。

## 8. 训练

使用下面命令进行启动训练（训练大约需要 2.5 个小时）：

```shell
python tools/train.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py
```

下面是 `1 x 3080Ti`、`batch size = 32`，训练 `100 epoch` 最佳精度权重 `work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat/best_coco/bbox_mAP_epoch_100.pth` 得出来的精度（详细机器资料可见附录）：

```shell
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.950
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.950
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.869
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.964
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.964
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.964

bbox_mAP_copypaste: 0.950 1.000 1.000 -1.000 -1.000 0.950
Epoch(val) [100][116/116]  coco/bbox_mAP: 0.9500  coco/bbox_mAP_50: 1.0000  coco/bbox_mAP_75: 1.0000  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.9500
```

## 9. 推理

使用最佳的模型进行推理，下面命令中的最佳模型路径是 `./work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat/best_coco/bbox_mAP_epoch_100.pth`，请用户自行修改为自己训练的最佳模型路径。

```shell
python demo/image_demo.py ./data/cat/images \
                          ./configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                          ./work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat/best_coco/bbox_mAP_epoch_100.pth \
                          --out-dir ./data/cat/pred_images
```

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/204773727-5d3cbbad-1265-45a0-822a-887713555049.jpg" alt="推理图片"/>
</div>

**Tips**：如果推理结果不理想，这里举例 2 种情况：

1. 欠拟合：
   需要先判断是不是训练 epoch 不够导致的欠拟合，如果是训练不够，则修改 config 文件里面的 `max_epochs` 和 `work_dir` 参数，或者根据上面的命名方式新建一个 config 文件，重新进行训练。

2. 数据集优化：
   如果 epoch 加上去了还是不行，可以增加数据集数量，同时可以重新检查并优化数据集的标注，然后重新进行训练。

## 10. 部署

MMYOLO 提供两种部署方式：

1. [MMDeploy](https://github.com/open-mmlab/mmdeploy) 框架进行部署
2. 使用 `projects/easydeploy` 进行部署

### 8.1 MMDeploy 框架进行部署

详见[YOLOv5 部署全流程说明](https://mmyolo.readthedocs.io/zh_CN/latest/deploy/yolov5_deployment.html)

### 8.2 使用 `projects/easydeploy` 进行部署

详见[部署文档](https://github.com/open-mmlab/mmyolo/blob/dev/projects/easydeploy/README_zh-CN.md)

TODO: 下个版本会完善这个部分...

## 附录

### 1. 本教程训练机器的详细环境的资料如下：

```shell
sys.platform: linux
Python: 3.9.13 | packaged by conda-forge | (main, May 27 2022, 16:58:50) [GCC 10.3.0]
CUDA available: True
numpy_random_seed: 2147483648
GPU 0: NVIDIA GeForce RTX 3080 Ti
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.5, V11.5.119
GCC: gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
PyTorch: 1.10.0
PyTorch compiling details: PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) oneAPI Math Kernel Library Version 2021.4-Product Build 20210904 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.2.3 (Git Hash 7336ca9f055cf1bfa13efb658fe15dc9b41f0740)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.3
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;
                             arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;
                             -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;
                             arch=compute_86,code=sm_86;-gencode;arch=compute_37,code=compute_37
  - CuDNN 8.2
  - Magma 2.5.2
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.3, CUDNN_VERSION=8.2.0,
                    CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden
                    -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK
                    -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra
                    -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas
                    -Wno-sign-compare -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic
                    -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new
                    -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format
                    -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1,
                    TORCH_VERSION=1.10.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON,
                    USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON,

TorchVision: 0.11.0
OpenCV: 4.6.0
MMEngine: 0.3.1
MMCV: 2.0.0rc3
MMDetection: 3.0.0rc3
MMYOLO: 0.1.3+3815671
```
