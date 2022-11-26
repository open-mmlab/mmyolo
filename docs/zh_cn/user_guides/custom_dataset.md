# 自定义数据集 标注+训练+测试+部署 全流程

本章节会介绍从 用户自定义图片数据集标注 到 最终进行训练和部署 的整体流程。流程步骤概览如下：

1. 使用 [labelme](https://github.com/wkentaro/labelme) 进行数据集标注：`demo/image_demo.py` + labelme
2. 使用脚本转换成 COCO 数据集格式：`tools/dataset_converters/labelme2coco.py`
3. 数据集划分：`tools/misc/coco_split.py`
4. 根据数据集内容新建 config 文件
5. 训练：`tools/train.py`
6. 推理：`demo/image_demo.py`
7. 部署

下面详细介绍每一步。

## 1. 使用 labelme 进行数据集标注

通常，标注有 2 种方法：

- 软件或者算法辅助 + 人工修正标签
- 仅人工打标签

## 1.1 软件或者算法辅助 + 人工修正标签

辅助标注的原理是用已有模型进行推理，将得出的推理信息保存为标注软件的标签文件格式。

**Tips**：如果已有模型典型的如 COCO 预训练模型没有你自定义新数据集的类别，建议先人工打 100 张左右的图片标签，训练个初始模型，然后再进行辅助打标签。

人工操作标注软件加载生成好的标签文件，只需要检查每张图片的目标是否标准，以及是否有漏掉的目标。

【辅助 + 人工标注】这种方式可以节省很多时间和精力，达到降本提速的目的。

下面会分别介绍其过程：

### 1.1.1 软件或者算法辅助

MMYOLO 提供的模型推理脚本 `demo/image_demo.py` 设置 `--to-labelme` 可以生成 labelme 格式标签文件，具体用法如下：

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
- `--to-labelme`：是否导出 `labelme` 格式的标签文件，不可以与 `--show` 参数同时存在

例子：

这里使用 YOLOv5-s 作为例子来进行辅助标注，先下载 YOLOv5-s 的权重:

```shell
mkdir work_dirs
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth -P ./work_dirs
```

执行辅助标注有 2 种情况：

1. 标注文件保存所有推理出来的类
2. 标注文件只保存 cat、dog 两类

- 标注文件保存所有推理出来的类：

```shell
python demo/image_demo.py /data/cat/images \
                          configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                          work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
                          --out-dir /data/cat/labels \
                          --to-labelme
```

- 标注文件只保存 `cat`、`dog` 两类：

```shell
python demo/image_demo.py /data/cat/images \
                          configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                          work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
                          --out-dir /data/cat/labels \
                          --class-name cat dog \
                          --to-labelme
```

生成的标签文件会在 `--out-dir` 中:

```shell
.
└── $OUT_DIR
    ├── image1.json
    ├── image1.json
    └── ...
```

### 1.1.2 人工标注

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
labelme /data/cat/images --output /data/cat/labels --autosave --nodata
```

输入命令之后 labelme 就会启动，然后进行标签检查即可。如果 labelme 启动失败，命令行输入 `export QT_DEBUG_PLUGINS=1` 查看具体缺少什么库，安装一下即可。

**注意：标注的时候务必使用 `rectangle`，快捷键 `Ctrl + R`（如下图）**

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/204076212-86dab4fa-13dd-42cd-93d8-46b04b864449.png" alt="rectangle"/>
</div>

## 1.2 仅人工打标签

步骤和 【1.1.2 人工标注】 相同，只是这里是直接标注，没有预先生成的标签。

## 2. 使用脚本转换成 COCO 数据集格式

### 2.1 使用脚本转换

MMYOLO 提供脚本将 labelme 的标签转换为 COCO 标签

```shell
python tools/dataset_converters/labelme2coco.py --img-dir ${图片文件夹路径} \
                                                --labels-dir ${标签文件夹位置} \
                                                --out ${输出 COCO 标签json路径}
                                                [--class-id-txt]
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

### 2.2 检查转换的 COCO 标签

使用下面的命令可以将 COCO 的标签在图片上进行显示，这一步可以验证刚刚转换是否有问题：

```shell
python tools/analysis_tools/browse_coco_json.py --img-dir ${图片文件夹路径} \
                                                --ann-file ${COCO 标签json路径}
```

关于 `tools/analysis_tools/browse_coco_json.py` 的更多用法请参考 [可视化 COCO 标签](useful_tools.md)。

## 3. 数据集划分

```shell
python tools/misc/coco_split.py --json ${COCO 标签 json 路径} \
                                --out-dir ${划分标签 json 保存根路径} \
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

## 4. 根据数据集内容新建 config 文件

确保数据集目录是这样的：

```shell
.
└── $DATA_ROOT
    ├── annotations
    │    ├── train.json # or trainval.json
    │    ├── val.json # optional
    │    └── test.json
    ├── images
    │    ├── a.jpg
    │    ├── b.png
    │    └── ...
    └── ...
```

因为是我们自定义的数据集，所以我们需要自己重写 config 中的部分信息，我们在 configs 目录下新建一个新的目录 `custom_dataset`，同时新建 config 文件。
这个 config 继承的是 `yolov5_s-v61_syncbn_8xb16-300e_coco.py`，假设数据集中的类是猫，`batch size = 8`，`200 epoch`，可以将其命名为 `yolov5_s-v61_syncbn_fast_1xb8-200e_cat.py`，并在其里面添加以下内容：

```python
_base_ = '../yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'

max_epochs = 200  # 训练的最大 epoch
data_root = '/path/to/data_root/'  # 数据集目录的绝对路径

# 结果保存的路径，如果同个 config 只是修改了部分参数，修改这个变量就可以将新的训练文件保存到其他地方
work_dir = './work_dirs/yolov5_s-v61_syncbn_fast_1xb8-200e_cat'

# checkpoint 可以指定本地路径或者 URL，设置了 URL 会自动进行下载，因为上面已经下载过，我们这里设置本地路径
checkpoint = './work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'

resume = False  # 继续训练的时候使用

train_batch_size_per_gpu = 32  # 根据自己的 GPU 情况修改 batch size，YOLOv5-s 默认为 16 * 8
train_num_workers = 4  # 推荐使用 train_num_workers = nGPU x 4
val_batch_size_per_gpu = 2  # val 时候的 batch size ，根据实际调整即可
val_num_workers = 2

save_epoch_intervals = 2  # 每 interval 轮迭代进行一次保存一次权重

# 根据自己的 GPU 情况，修改 base_lr，修改的比例是 base_lr_default * (your_bs / default_bs)
base_lr = _base_.base_lr / 4  

num_classes = 1
metainfo = dict(  # 根据 class_with_id.txt 类别信息，设置 metainfo
    CLASSES=('cat'),
    PALETTE=[(220, 20, 60)]  # 画图时候的颜色，随便设置即可
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=1,  # 第几个epoch后验证
    val_interval=save_epoch_intervals  # 每 val_interval 轮迭代进行一次测试评估
)

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    bbox_head=dict(head_module=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/trainval.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/trainval.json',
        data_prefix=dict(img='images/')))

test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))

val_evaluator = dict(ann_file=data_root + 'annotations/trainval.json')
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=base_lr))

default_hooks = dict(
    # 设置间隔多少个 epoch 保存模型，以及保存模型最多几个，`save_best` 是另外保存最佳模型（推荐）
    checkpoint=dict(type='CheckpointHook', interval=save_epoch_intervals, 
                    max_keep_ckpts=5, save_best='auto'),
    # logger 输出的间隔
    logger=dict(type='LoggerHook', interval=5)
)
```

## 5. 训练

使用下面命令进行启动训练：

```shell
python tools/train.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb8-200e_cat.py
```

## 6. 推理

```shell
python demo/image_demo.py /path/to/test/images \
                          configs/my_yolov5_s_config.py \
                          ./work_dirs/yolov5_s-v61_syncbn_fast_1xb8-200e_cat/yolov5_s-v61_syncbn_fast_1xb8-200e_cat/last.pth \
                          --out-dir /path/to/test/images_output
```

**Tips**：如果推理结果不理想，这里举例 2 中情况：

1. 欠拟合：
   需要先判断是不是训练 epoch 不够导致的欠拟合，如果是训练不够，则修改 config 文件里面的 `resume = ./work_dirs/yolov5_s-v61_syncbn_fast_1xb8-200e_cat/yolov5_s-v61_syncbn_fast_1xb8-200e_cat/last.pth` 来最后的模型继续训练。

2. 数据集优化：
   如果 epoch 加上去了还是不行，可以对数据集进行增加，同时加上不断修改标注来优化，然后重新进行训练。

## 7. 部署