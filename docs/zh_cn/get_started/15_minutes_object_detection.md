# 15 分钟上手 MMYOLO 目标检测

以我们提供的猫 cat 小数据集为例，带大家 15 分钟轻松上手 MMYOLO 目标检测。整个流程包含如下步骤：

- [环境安装](#环境安装)
- [数据集准备](#数据集准备)
- [配置准备](#配置准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [EasyDeploy 模型部署](#easydeploy-模型部署)

## 环境安装

假设你已经提前安装好了 Conda，接下来安装 PyTorch

```shell
conda create -n mmyolo python=3.8 -y
conda activate mmyolo
conda install pytorch torchvision -c pytorch
# conda install pytorch torchvision cpuonly -c pytorch
```

安装 MMYOLO 和依赖库

```shell
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
pip install -U openmim
mim install -r requirements/mminstall.txt
# Install albumentations
mim install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

```{note}
温馨提醒：由于本仓库采用的是 OpenMMLab 2.0，请最好新建一个 conda 虚拟环境，防止和 OpenMMLab 1.0 已经安装的仓库冲突。
```

详细环境配置操作请查看 [安装和验证](./installation.md)

## 数据集准备

Cat 数据集是一个包括 144 张图片的单类别数据集（本 cat 数据集由 @RangeKing 提供原始图片，由 @PeterH0323 进行数据清洗）, 包括了训练所需的标注信息。 样例图片如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205423220-c4b8f2fd-22ba-4937-8e47-1b3f6a8facd8.png" alt="cat dataset"/>
</div>

你只需执行如下命令即可下载并且直接用起来

```shell
python tools/misc/download_dataset.py --dataset-name cat --save-dir ./data/cat --unzip --delete
```

数据集组织格式如下所示：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/220072078-48b88a08-6179-483e-b8d3-0549e1b465de.png" alt="image"/>
</div>

data 位于 mmyolo 工程目录下， `data/cat/annotations` 中存放的是 COCO 格式的标注，`data/cat/images` 中存放的是所有图片

## 配置准备

以 YOLOv5 算法为例，考虑到用户显存和内存有限， 我们需要修改一些默认训练参数来让大家愉快的跑起来，核心需要修改的参数如下

- YOLOv5 是 Anchor-Based 类算法，不同的数据集需要自适应计算合适的 Anchor
- 默认配置是 8 卡，每张卡 batch size 为 16，现将其改成单卡，每张卡 batch size 为 8
- 默认训练 epoch 是 300，先将其改成 50 epoch

具体操作为在 `configs/yolov5` 文件夹下新建 `yolov5_s-v61_fast_1xb8-50e_cat.py` 配置文件(为了方便大家直接使用，我们已经提供了该配置)，并把以下内容复制配置文件中。

```python
# TODO 加注释
_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = './data/cat/'
class_name = ('cat', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]

max_epochs = 50
train_batch_size_per_gpu = 8
train_num_workers = 4
# base_lr_default * (your_bs / default_bs)
base_lr = _base_.base_lr / 4  # TODO

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/trainval.json',
        data_prefix=dict(img='images/')))

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
    checkpoint=dict(interval=5, max_keep_ckpts=2, save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=10))
train_cfg = dict(
    max_epochs=max_epochs, val_interval=5)
```

以上配置从 `yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py` 中继承，并根据 cat 数据的特点更新了 `data_root`、`metainfo`、`train_dataloader`、`val_dataloader`、`num_classes` 等配置。

## 模型训练

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_fast_1xb8-50e_cat.py
```

运行以上训练命令，`work_dirs/yolov5_s-v61_fast_1xb8-50e_cat` 文件夹会被自动生成，权重文件以及此次的训练配置文件将会保存在此文件夹中。

<div align=center>
<img src="https://cdn.vansin.top/img/20220913213846.png" alt="image"/>
</div>

### 中断后恢复训练

如果训练中途停止，可以在训练命令最后加上 `--resume` ,程序会自动从 `work_dirs` 中加载最新的权重文件恢复训练。

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_fast_1xb8-50e_cat.py --resume
```

### 节省显存策略

上述配置大概需要 4.5G 显存，如果你的显存不够，可以考虑开启混合精度训练

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_fast_1xb8-50e_cat.py --amp
```

### 训练可视化

MMYOLO 目前支持本地、TensorBoard 以及 WandB 等多种后端可视化，默认是采用本地可视化方式，你可以切换为 WandB 等实时可视化训练过程中各类指标。

##### 1 WandB 可视化使用

WandB 官网注册并在 https://wandb.ai/settings 获取到 WandB 的 API Keys。

<div align=center>
<img src="https://cdn.vansin.top/img/20220913212628.png" alt="image"/>
</div>

```shell
pip install wandb
# 运行了 wandb login 后输入上文中获取到的 API Keys ，便登录成功。
wandb login
```

在 `configs/yolov5/yolov5_s-v61_fast_1xb8-50e_cat.py` 添加 WandB 配置

```python
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
```

重新运行训练命令便可以在命令行中提示的网页链接中看到 loss、学习率和 coco/bbox_mAP 等数据可视化了。

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_fast_1xb8-50e_cat.py
```

<div align=center>
<img src="https://cdn.vansin.top/img/20220913213221.png" alt="image"/>
</div>

##### 2 Tensorboard 可视化使用

安装 Tensorboard 环境

```shell
pip install tensorboard
```

同上述在配置文件 `configs/yolov5/yolov5_s-v61_fast_1xb8-50e_cat.py`中添加 `tensorboard` 配置

```python
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
```

重新运行训练命令后，Tensorboard 文件会生成在可视化文件夹 `work_dirs/yolov5_s-v61_fast_1xb8-50e_cat.py/{timestamp}/vis_data` 下，
运行下面的命令便可以在网页链接使用 Tensorboard 查看 loss、学习率和 coco/bbox_mAP 等可视化数据了：

```shell
tensorboard --logdir=work_dirs/yolov5_s-v61_fast_1xb8-50e_cat.py
```

## 模型测试

```shell
python tools/test.py configs/yolov5/yolov5_s-v61_fast_1xb8-50e_cat.py \
                     work_dirs/yolov5_s-v61_fast_1xb8-50e_cat/epoch_50.pth \
                     --show-dir show_results
```

运行以上测试命令，推理结果图片会自动保存至 `work_dirs/yolov5_s-v61_fast_1xb8-50e_cat/{timestamp}/show_results` 文件夹中。下面为其中一张结果图片，左图为实际标注，右图为模型推理结果。

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/190913272-f99709e5-c798-46b8-aede-30f4e91683a3.jpg" alt="result_img"/>
</div>

## EasyDeploy 模型部署

TODO

以上完整内容可以查看 \[15_minutes_object_detection.ipynb\]
