# 自定义数据集 标注+训练+测试+部署 全流程

在平时的工作学习中，我们经常会遇到一些任务需要训练自定义的私有数据集，开源数据集去作为上线模型的场景比较少，这就需要我们对自己的私有数据集进行一系列的操作，以确保模型能够上线生产服务于客户。

```{note}
本教程所有指令是在 Linux 上面完成，Windows 也是完全可用的，但是命令和操作稍有不同。
```

本教程默认您已经完成 MMYOLO 的安装，如果未安装，请参考文档 [开始你的第一步](https://mmyolo.readthedocs.io/zh_CN/latest/get_started.html#id1) 进行安装。

本章节会介绍从 用户自定义图片数据集标注 到 最终进行训练和部署 的整体流程。流程步骤概览如下：

01. 数据集准备：`tools/misc/download_dataset.py`
02. 使用 [labelme](https://github.com/wkentaro/labelme) 和算法进行辅助和优化数据集标注：`demo/image_demo.py` + labelme
03. 使用脚本转换成 COCO 数据集格式：`tools/dataset_converters/labelme2coco.py`
04. 数据集划分为训练集、验证集和测试集：`tools/misc/coco_split.py`
05. 根据数据集内容新建 config 文件
06. 数据集可视化分析：`tools/analysis_tools/dataset_analysis.py`
07. 优化 Anchor 尺寸：`tools/analysis_tools/optimize_anchors.py`
08. 可视化 config 配置中数据处理部分： `tools/analysis_tools/browse_dataset.py`
09. 训练：`tools/train.py`
10. 推理：`demo/image_demo.py`
11. 部署

```{note}
在训练得到模型权重和验证集的 mAP 后，用户会需要对预测错误的情况进行深入分析，以便优化模型，MMYOLO 在后续会增加这个功能，敬请期待。
```

下面详细介绍每一步。

## 1. 数据集准备

- 如果您现在暂时没有自己的数据集，亦或者想尝试用一个小型数据集来跑通我们的 demo，可以使用本教程提供的一个 144 张图片的 `cat` 数据集（本 `cat` 数据集由 @RangeKing 提供原始图片，由 @PeterH0323 进行数据清洗）。本教程的剩余部分都将以此 `cat` 数据集为例进行讲解。

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205423220-c4b8f2fd-22ba-4937-8e47-1b3f6a8facd8.png" alt="cat dataset"/>
</div>

下载也非常简单，只需要一条命令即可完成：

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

```{note}
这个数据集可以直接训练，如果您想体验整个流程的话，可以将 `images` 文件夹**以外的**其余文件都删除。
```

- 如您已经有数据，可以将其组成下面的结构：

```shell
.
└── $DATA_ROOT
    └── images
         ├── image1.jpg
         ├── image2.png
         └── ...
```

## 2. 使用 labelme 和算法进行辅助和优化数据集标注

通常，标注有 2 种方法：

- 软件或者算法辅助 + 人工修正 label（推荐，降本提速）
- 仅人工标注

目前我们也在考虑接入第三方库来支持通过 GUI 界面调用 MMYOLO 推理接口实现算法辅助标注和人工优化标注一体功能。如果您有兴趣或者想法可以在 issue 留言或直接联系我们！

## 2.1 软件或者算法辅助 + 人工修正 label

辅助标注的原理是用已有模型进行推理，将得出的推理信息保存为标注软件 label 文件格式。然后人工操作标注软件加载生成好的 label 文件，只需要检查每张图片的目标是否标准，以及是否有漏掉的目标。【辅助 + 人工标注】这种方式可以节省很多时间和精力，达到**降本提速**的目的。

```{note}
如果已有模型（典型的如 COCO 预训练模型）没有您自定义新数据集的类别，建议先人工打 100 张左右的图片 label，训练个初始模型，然后再进行辅助标注。
```

下面会分别介绍其过程：

### 2.1.1 软件或者算法辅助

MMYOLO 提供的模型推理脚本 `demo/image_demo.py` 设置 `--to-labelme` 可以将推理结果生成 labelme 格式的 label 文件，具体用法如下：

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

- 如果您的数据集需要标注多类，可以采用类似 `--class-name class1 class2` 格式输入；
- 如果全部输出，则删掉 `--class-name` 这个 flag 即可全部类都输出。

生成的 label 文件会在 `--out-dir` 中:

```shell
.
└── $OUT_DIR
    ├── image1.json
    ├── image1.json
    └── ...
```

这是一张原图及其生成的 json 例子：

<div align=center>
  <img src="https://user-images.githubusercontent.com/25873202/205471430-dcc882dd-16bb-45e4-938f-6b62ab3dff19.jpg" alt="图片" width="45%"/>
  <img src="https://user-images.githubusercontent.com/25873202/205471559-643aecc8-7fa3-4fff-be51-2fb0a570fdd3.png" alt="图片" width="45%"/>
</div>

### 2.1.2 人工标注

本教程使用的标注软件是 [labelme](https://github.com/wkentaro/labelme)

- 安装 labelme

```shell
conda create -n labelme python=3.8
conda activate labelme
pip install labelme==5.1.1
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
cd ${MMYOLO_PATH}
labelme ./data/cat/images --output ./data/cat/labels --autosave --nodata
```

输入命令之后 labelme 就会启动，然后进行 label 检查即可。如果 labelme 启动失败，命令行输入 `export QT_DEBUG_PLUGINS=1` 查看具体缺少什么库，安装一下即可。

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205432185-54407d83-3cee-473f-8743-656da157cf80.png" alt="label UI"/>
</div>

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

以本教程的 `cat` 数据集为例：

```shell
python tools/dataset_converters/labelme2coco.py --img-dir ./data/cat/image \
                                                --labels-dir ./data/cat/labels \
                                                --out ./data/cat/annotations/annotations_all.json
```

本次演示的 `cat` 数据集（注意不需要包括背景类），可以看到生成的 `class_with_id.txt` 中只有 `1` 类：

```text
1 cat

```

### 3.2 检查转换的 COCO label

使用下面的命令可以将 COCO 的 label 在图片上进行显示，这一步可以验证刚刚转换是否有问题：

```shell
python tools/analysis_tools/browse_coco_json.py --img-dir ${图片文件夹路径} \
                                                --ann-file ${COCO label json 路径}
```

例子：

```shell
python tools/analysis_tools/browse_coco_json.py --img-dir ./data/cat/images \
                                                --ann-file ./data/cat/annotations/annotations_all.json
```

<div align=center>
<img alt="Image" src="https://user-images.githubusercontent.com/25873202/205429166-a6e48d20-c60b-4571-b00e-54439003ad3b.png">
</div>

关于 `tools/analysis_tools/browse_coco_json.py` 的更多用法请参考 [可视化 COCO label](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#coco)。

## 4. 数据集划分为训练集、验证集和测试集

通常，自定义图片都是一个大文件夹，里面全部都是图片，需要我们自己去对图片进行训练集、验证集、测试集的划分，如果数据量比较少，可以不划分验证集。下面是划分脚本的具体用法：

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
                                --seed 10
```

<div align=center>
<img alt="Image" src="https://user-images.githubusercontent.com/25873202/205428346-5fdfbfca-0682-47aa-b0be-fa467cd0c5f8.png">
</div>

## 5. 根据数据集内容新建 config 文件

确保数据集目录是这样的：

```shell
.
└── $DATA_ROOT
    ├── annotations
    │    ├── trainval.json # 根据上面的指令只划分 trainval + test，如果您使用 3 组划分比例的话，这里是 train.json、val.json、test.json
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

<div align=center>
<img alt="Image" src="https://user-images.githubusercontent.com/25873202/205428358-e32fb455-480a-4f14-9613-e4cc3193fb4d.png">
</div>

```python
_base_ = '../yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

max_epochs = 100  # 训练的最大 epoch
data_root = './data/cat/'  # 数据集目录的绝对路径
# data_root = '/root/workspace/mmyolo/data/cat/'  # Docker 容器里面数据集目录的绝对路径

# 结果保存的路径，可以省略，省略保存的文件名位于 work_dirs 下 config 同名的文件夹中
# 如果某个 config 只是修改了部分参数，修改这个变量就可以将新的训练文件保存到其他地方
work_dir = './work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat'

# load_from 可以指定本地路径或者 URL，设置了 URL 会自动进行下载，因为上面已经下载过，我们这里设置本地路径
# 因为本教程是在 cat 数据集上微调，故这里需要使用 `load_from` 来加载 MMYOLO 中的预训练模型，这样可以在加快收敛速度的同时保证精度
load_from = './work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

# 根据自己的 GPU 情况，修改 batch size，YOLOv5-s 默认为 8卡 x 16bs
train_batch_size_per_gpu = 32
train_num_workers = 4  # 推荐使用 train_num_workers = nGPU x 4

save_epoch_intervals = 2  # 每 interval 轮迭代进行一次保存一次权重

# 根据自己的 GPU 情况，修改 base_lr，修改的比例是 base_lr_default * (your_bs / default_bs)
base_lr = _base_.base_lr / 4

anchors = [  # 此处已经根据数据集特点更新了 anchor，关于 anchor 的生成，后面小节会讲解
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
    val_begin=20,  # 第几个 epoch 后验证，这里设置 20 是因为前 20 个 epoch 精度不高，测试意义不大，故跳过
    val_interval=save_epoch_intervals  # 每 val_interval 轮迭代进行一次测试评估
)

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),

        # loss_cls 会根据 num_classes 动态调整，但是 num_classes = 1 的时候，loss_cls 恒为 0
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

```{note}
我们在 `projects/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py` 放了一份相同的 config 文件，用户可以选择复制到 `configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py` 路径直接开始训练。
```

## 6. 数据集可视化分析

脚本 `tools/analysis_tools/dataset_analysis.py` 能够帮助用户得到数据集的分析图。该脚本可以生成 4 种分析图：

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

以本教程 `cat` 数据集 的 config 为例：

查看训练集数据分布情况：

```shell
python tools/analysis_tools/dataset_analysis.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                                                --output-dir work_dirs/dataset_analysis_cat/train_dataset
```

查看验证集数据分布情况：

```shell
python tools/analysis_tools/dataset_analysis.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                                                --output-dir work_dirs/dataset_analysis_cat/val_dataset \
                                                --val-dataset
```

效果（点击图片可查看大图）：

<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>基于面积规则下，显示类别和 bbox 实例面积的分布图</b>
      </td>
      <td>
        <b>显示类别和 bbox 实例宽、高的分布图</b>
      </td>
    </tr>
    <tr align="center" valign="center">
      <td>
        <img alt="YOLOv5CocoDataset_bbox_area" src="https://user-images.githubusercontent.com/25873202/205805522-e066b93b-0952-40d0-be56-42fc20e85576.jpg" width="60%">
      </td>
      <td>
        <img alt="YOLOv5CocoDataset_bbox_wh" src="https://user-images.githubusercontent.com/25873202/205805514-13e34d18-f9ee-4bca-b894-060bfea6ad56.jpg" width="60%">
      </td>
    </tr>
    <tr align="center" valign="center">
      <td>
        <b>显示类别和 bbox 实例个数的分布图</b>
      </td>
      <td>
        <b>显示类别和 bbox 实例宽/高比例的分布图</b>
      </td>
    </tr>
    <tr align="center" valign="center">
      <td>
        <img alt="YOLOv5CocoDataset_bbox_num" src="https://user-images.githubusercontent.com/25873202/205805529-6e0d2545-0a39-4be7-b212-9d16a16a0fcc.jpg" width="60%">
      </td>
      <td>
        <img alt="YOLOv5CocoDataset_bbox_ratio" src="https://user-images.githubusercontent.com/25873202/205805501-2b2a40f4-5a0e-44b0-b07c-27af539fb971.jpg" width="60%">
      </td>
    </tr>
  </tbody>
</table>

```{note}
因为本教程使用的 cat 数据集数量比较少，故 config 里面用了 RepeatDataset，显示的数目实际上都是重复了 5 次。如果您想得到无重复的分析结果，可以暂时将 RepeatDataset 下面的 `times` 参数从 `5` 改成 `1`。
```

经过输出的图片分析可以得出，本教程使用的 cat 数据集的训练集具有以下情况：

- 图片全部是 large object；
- 类别 cat 的数量是 `129`；
- bbox 的宽高比例大部分集中在 `1.14`，比例最小值是 `0.36`，最大值是 `2.9`；
- bbox 的宽大部分是 `1034.74`，高大部分是 `926.67`。

关于 `tools/analysis_tools/dataset_analysis.py` 的更多用法请参考 [可视化数据集分析](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#id4)。

## 7. 优化 Anchor 尺寸

脚本 `tools/analysis_tools/optimize_anchors.py` 支持 YOLO 系列中三种锚框生成方式，分别是 `k-means`、`differential_evolution`、`v5-k-means`.

本示例使用的是 YOLOv5 进行训练，使用的是 `640 x 640` 的输入大小，使用 `v5-k-means` 进行描框的优化：

```shell
python tools/analysis_tools/optimize_anchors.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                                                --algorithm v5-k-means \
                                                --input-shape 640 640 \
                                                --prior-match-thr 4.0 \
                                                --out-dir work_dirs/dataset_analysis_cat
```

经过计算的 Anchor 如下：

<div align=center>
<img alt="Anchor" src="https://user-images.githubusercontent.com/25873202/205422434-1a68cded-b055-42e9-b01c-3e51f8f5ef81.png">
</div>

修改 config 文件里面的 `anchors` 变量：

```python
anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]
```

关于 `tools/analysis_tools/optimize_anchors.py` 的更多用法请参考 [优化锚框尺寸](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#id8)。

## 8. 可视化 config 配置中数据处理部分

脚本 `tools/analysis_tools/browse_dataset.py` 能够帮助用户去直接窗口可视化 config 配置中数据处理部分，同时可以选择保存可视化图片到指定文件夹内。

下面演示使用我们刚刚新建的 config 文件 `configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py` 来可视化图片，该命令会使得图片直接弹出显示，每张图片持续 `5` 秒，图片不进行保存：

```shell
python tools/analysis_tools/browse_dataset.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                                              --show-interval 5
```

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205472078-c958e90d-8204-4c01-821a-8b6a006f05b2.png" alt="image"/>
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205472197-8228c75e-6046-404a-89b4-ed55eeb2cb95.png" alt="image"/>
</div>

关于 `tools/analysis_tools/browse_dataset.py` 的更多用法请参考 [可视化数据集](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#id3)。

## 9. 训练

使用刚刚我们新建好的 config 文件执行训练。

### 9.1 训练可视化

如果需要采用浏览器对训练过程可视化，MMYOLO 目前提供 2 种方式 `[wandb](https://wandb.ai/site)` 和 `[TensorBoard](https://tensorflow.google.cn/tensorboard)`，根据自己的情况选择其一即可(后续会扩展更多可视化后端支持)。

#### 9.1.1 wandb

wandb 可视化需要在[官网](https://wandb.ai/site)注册，并在 https://wandb.ai/settings 获取到 wandb 的 API Keys。

<div align=center>
<img src="https://cdn.vansin.top/img/20220913212628.png" alt="image"/>
</div>

然后在命令行进行安装

```shell
pip install wandb
# 运行了 wandb login 后输入上文中获取到的 API Keys ，便登录成功。
wandb login
```

在我们刚刚新建的 config 文件 `configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py` 添加 wandb 配置：

```python
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
```

#### 9.1.2 TensorBoard

安装 Tensorboard 环境

```shell
pip install tensorboard
```

在我们刚刚新建的 config 文件 `configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py` 中添加 `tensorboard` 配置

```python
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
```

待会运行训练命令后，Tensorboard 文件会生成在可视化文件夹 `work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat/${TIMESTAMP}/vis_data` 下，
运行下面的命令便可以在网页链接使用 Tensorboard 查看 loss、学习率和 coco/bbox_mAP 等可视化数据了：

```shell
tensorboard --logdir=work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat
```

### 9.2 执行训练

使用下面命令进行启动训练（训练大约需要 2.5 个小时）：

```shell
python tools/train.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py
```

下面是 `1 x 3080Ti`、`batch size = 32`，训练 `100 epoch` 最佳精度权重 `work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat/best_coco/bbox_mAP_epoch_98.pth` 得出来的精度（详细机器资料可见附录）：

```shell
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.939
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.939
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.867
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.959
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.959
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.959

bbox_mAP_copypaste: 0.939 1.000 1.000 -1.000 -1.000 0.939
Epoch(val) [98][116/116]  coco/bbox_mAP: 0.9390  coco/bbox_mAP_50: 1.0000  coco/bbox_mAP_75: 1.0000  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.9390
```

在一般的 finetune 最佳实践中都会推荐将 backbone 固定不参与训练，并且学习率 lr 也进行相应缩放，但是在本教程中发现这种做法会出现一定程度掉点。猜测可能原因是 cat 类别已经在 COCO 数据集中，而 cat 数据集比较小。

## 10. 推理

使用最佳的模型进行推理，下面命令中的最佳模型路径是 `./work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat/best_coco/bbox_mAP_epoch_98.pth`，请用户自行修改为自己训练的最佳模型路径。

```shell
python demo/image_demo.py ./data/cat/images \
                          ./configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                          ./work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat/best_coco/bbox_mAP_epoch_98.pth \
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

## 11. 部署

MMYOLO 提供两种部署方式：

1. [MMDeploy](https://github.com/open-mmlab/mmdeploy) 框架进行部署
2. 使用 `projects/easydeploy` 进行部署

### 11.1 MMDeploy 框架进行部署

考虑到部署的机器环境千差万别，很多时候在本地机器可以，但是在生产环境则不一定，这里推荐使用 Docker，做到环境一次部署，终身使用，节省运维搭建环境和部署生产的时间。

1. 构建 Docker 镜像
2. 创建 Docker 容器
3. 转换 TensorRT 模型
4. 部署模型执行推理

如果是对 Docker 不熟悉的用户，可以参考 MMDeploy 的 [源码手动安装](https://mmdeploy.readthedocs.io/zh_CN/latest/01-how-to-build/build_from_source.html) 文档直接在本地编译。安装完之后，可以直接跳到 【11.1.3 转换 TensorRT 模型】 小节。

#### 11.1.1 构建 Docker 镜像

```bash
git clone -b dev-1.x https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
docker build docker/GPU/ -t mmdeploy:gpu --build-arg USE_SRC_INSIDE=true
```

其中 `USE_SRC_INSIDE=true` 是拉取基础进行之后在内部切换国内源，构建速度会快一些。

执行脚本后，会进行构建，此刻需要等一段时间：

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205482447-329186c8-eba3-443f-b1fa-b33c2ab3d5da.png" alt="Image"/>
</div>

#### 11.1.2 创建 Docker 容器

```shell
export MMYOLO_PATH=/path/to/local/mmyolo # 先将您机器上 MMYOLO 的路径写入环境变量
docker run --gpus all --name mmyolo-deploy -v ${MMYOLO_PATH}:/root/workspace/mmyolo -it mmdeploy:gpu /bin/bash
```

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205536974-1eeb2901-9b14-4851-9c96-5046cd05f171.png" alt="Image"/>
</div>

可以看到本地的 MMYOLO 环境已经挂载到容器里面了

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205537473-0afc16c3-c6d4-451a-96d7-1a2388341b60.png" alt="Image"/>
</div>

有关这部分的详细介绍可以看 MMDeploy 官方文档 [使用 Docker 镜像](https://mmdeploy.readthedocs.io/zh_CN/latest/01-how-to-build/build_from_docker.html#docker)

#### 11.1.3 转换 TensorRT 模型

首先需要在 Docker 容器里面安装 MMYOLO 和 `pycuda`：

```shell
export MMYOLO_PATH=/root/workspace/mmyolo # 镜像中的路径，这里不需要修改
cd ${MMYOLO_PATH}
export MMYOLO_VERSION=$(python -c "import mmyolo.version as v; print(v.__version__)")  # 查看训练使用的 MMYOLO 版本号
echo "Using MMYOLO ${MMYOLO_VERSION}"
mim install --no-cache-dir mmyolo==${MMYOLO_VERSION}
pip install --no-cache-dir pycuda==2022.2
```

进行模型转换

```shell
cd /root/workspace/mmdeploy
python ./tools/deploy.py \
    ${MMYOLO_PATH}/configs/deploy/detection_tensorrt-fp16_dynamic-192x192-960x960.py \
    ${MMYOLO_PATH}/configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
    ${MMYOLO_PATH}/work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat/best_coco/bbox_mAP_epoch_98.pth \
    ${MMYOLO_PATH}/data/cat/images/mmexport1633684751291.jpg \
    --test-img ${MMYOLO_PATH}/data/cat/images/mmexport1633684751291.jpg \
    --work-dir ./work_dir/yolov5_s-v61_syncbn_fast_1xb32-100e_cat_deploy_dynamic_fp16 \
    --device cuda:0 \
    --log-level INFO \
    --show \
    --dump-info
```

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205540259-ded15231-c428-4a5b-ac45-06cf15c5b7e9.png" alt="Image"/>
</div>

等待一段时间，出现了 `All process success.` 即为成功：

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205540981-355d34cb-6472-47e0-a7dd-11eb85b3b43c.png" alt="Image"/>
</div>

查看导出的路径，可以看到这样的文件结构：

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205541268-1807f3fd-1f22-42b0-9397-cb50c602f8e0.png" alt="Image"/>
</div>

关于转换模型的详细介绍，请参考 [如何转换模型](https://mmdeploy.readthedocs.io/zh_CN/latest/02-how-to-run/convert_model.html)

#### 11.1.4 部署模型执行推理

需要将 `${MMYOLO_PATH}/configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py` 里面的 `data_root` 修改为 Docker 容器里面的路径：

```python
data_root = '/root/workspace/mmyolo/data/cat/'  # Docker 容器里面数据集目录的绝对路径
```

可以执行速度和精度测试：

```shell
python tools/test.py \
    ${MMYOLO_PATH}/configs/deploy/detection_tensorrt-fp16_dynamic-192x192-960x960.py \
    ${MMYOLO_PATH}/configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
    --model ./work_dir/yolov5_s-v61_syncbn_fast_1xb32-100e_cat_deploy_dynamic_fp16/end2end.engine \
    --speed-test \
    --device cuda
```

速度测试如下，可见平均推理速度是 `18.31 ms`，对比 PyTorch 推理有速度提升，同时显存也下降了很多：

```bash
Epoch(test) [ 10/116]    eta: 0:00:10  time: 0.0950  data_time: 0.0844  memory: 12
Epoch(test) [ 20/116]    eta: 0:00:09  time: 0.0945  data_time: 0.0891  memory: 12
Epoch(test) [ 30/116]    eta: 0:00:08  time: 0.0953  data_time: 0.0804  memory: 12
Epoch(test) [ 40/116]    eta: 0:00:07  time: 0.0902  data_time: 0.0712  memory: 12
Epoch(test) [ 50/116]    eta: 0:00:06  time: 0.0858  data_time: 0.0622  memory: 12
Epoch(test) [ 60/116]    eta: 0:00:05  time: 0.0902  data_time: 0.0662  memory: 12
Epoch(test) [ 70/116]    eta: 0:00:04  time: 0.0901  data_time: 0.0645  memory: 16
Epoch(test) [ 80/116]    eta: 0:00:03  time: 0.0761  data_time: 0.0507  memory: 12
Epoch(test) [ 90/116]    eta: 0:00:02  time: 0.0958  data_time: 0.0692  memory: 12
Epoch(test) [100/116]    eta: 0:00:01  time: 0.0904  data_time: 0.0571  memory: 12
[tensorrt]-110 times per count: 18.31 ms, 54.61 FPS
Epoch(test) [110/116]    eta: 0:00:00  time: 0.1123  data_time: 0.0896  memory: 12
```

精度测试如下。此配置采用 FP16 格式推理，会有一定程度掉点，但是推理速度更快：

```shell
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.937
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.937
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.865
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.957
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.957
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.957

bbox_mAP_copypaste: 0.937 1.000 1.000 -1.000 -1.000 0.937
Epoch(test) [116/116]  coco/bbox_mAP: 0.9370  coco/bbox_mAP_50: 1.0000  coco/bbox_mAP_75: 1.0000  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.9370
```

单张图片推理：

```shell
cd ${MMYOLO_PATH}/demo
python deploy_demo.py \
    ${MMYOLO_PATH}/data/cat/images \
    --model-cfg ${MMYOLO_PATH}/configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
    --backend-model /root/workspace/mmdeploy/work_dir/yolov5_s-v61_syncbn_fast_1xb32-100e_cat_deploy_dynamic_fp16/end2end.engine \
    --deploy-cfg ${MMYOLO_PATH}/configs/deploy/detection_tensorrt-fp16_dynamic-192x192-960x960.py \
    --out-dir ${MMYOLO_PATH}/work_dirs/deploy_predict_out \
    --device cuda:0
```

执行之后，可以看到在 `--out-dir` 下面的推理图片结果，下面展示其中一张：

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205815829-6f85e655-722a-47c8-9e23-2a74437c0923.jpg" alt="Image"/>
</div>

```{note}
您也可以做其他优化调整，例如增大 batch，量化 int8 等等。
```

#### 11.1.4 保存和加载 Docker 容器

因为如果每次都进行 docker 镜像的构建，特别费时间，此时你可以考虑使用 docker 自带的打包 api 进行打包和加载。

```shell
# 保存，存好的 tar 包可以放到移动硬盘
docker save mmyolo-deploy > mmyolo-deploy.tar

# 加载
docker load < /path/to/mmyolo-deploy.tar
```

### 11.2 使用 `projects/easydeploy` 进行部署

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
MMYOLO: 0.2.0+cf279a5
```
