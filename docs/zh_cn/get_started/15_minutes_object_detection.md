# 15 分钟上手 MMYOLO 目标检测

## 环境安装

温馨提醒：由于本仓库采用的是 OpenMMLab 2.0，请最好新建一个 conda 虚拟环境，防止和 OpenMMLab 1.0 已经安装的仓库冲突。

```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch
# conda install pytorch torchvision cpuonly -c pytorch
pip install -U openmim
mim install "mmengine>=0.3.1"
mim install "mmcv>=2.0.0rc1,<2.1.0"
mim install "mmdet>=3.0.0rc5,<3.1.0"
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```

详细环境配置操作请查看 [get_started](../get_started/installation.md)

## 数据集准备

本文选取不到 40MB 大小的 balloon 气球数据集作为 MMYOLO 的学习数据集。

```shell
python tools/misc/download_dataset.py --dataset-name balloon --save-dir data --unzip
python tools/dataset_converters/balloon2coco.py
```

执行以上命令，下载数据集并转化格式后，balloon 数据集在 `data` 文件夹中准备好了，`train.json` 和 `val.json` 便是 coco 格式的标注文件了。

<div align=center>
<img src="https://cdn.vansin.top/img/20220912105312.png" alt="image"/>
</div>

## config 文件准备

在 `configs/yolov5` 文件夹下新建 `yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py` 配置文件，并把以下内容复制配置文件中。

```python
_base_ = './yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = 'data/balloon/'

train_batch_size_per_gpu = 4
train_num_workers = 2

metainfo = {
    'classes': ('balloon', ),
    'palette': [
        (220, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='val/'),
        ann_file='val.json'))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val.json')

test_evaluator = val_evaluator

model = dict(bbox_head=dict(head_module=dict(num_classes=1)))

default_hooks = dict(logger=dict(interval=1))
```

以上配置从 `./yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py` 中继承，并根据 balloon 数据的特点更新了 `data_root`、`metainfo`、`train_dataloader`、`val_dataloader`、`num_classes` 等配置。
我们将 logger 的 `interval` 设置为 1 的原因是，每进行 `interval` 次 iteration 会输出一次 loss 相关的日志，而我们选取气球数据集比较小，`interval` 太大我们将看不到 loss 相关日志的输出。

## 训练

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py
```

运行以上训练命令，`work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon` 文件夹会被自动生成，权重文件以及此次的训练配置文件将会保存在此文件夹中。

<div align=center>
<img src="https://cdn.vansin.top/img/20220913213846.png" alt="image"/>
</div>

### 中断后恢复训练

如果训练中途停止，在训练命令最后加上 `--resume` ,程序会自动从 `work_dirs` 中加载最新的权重文件恢复训练。

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py --resume
```

### 加载预训练权重微调

经过测试，相比不加载预训练模型，加载 YOLOv5-s 预训练模型在气球数据集上训练和验证 coco/bbox_mAP 能涨 30 多个百分点。

1. 下载 COCO 数据集预训练权重

```shell
cd mmyolo
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth
```

2. 加载预训练模型进行训练

```shell
cd mmyolo
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py \
                      --cfg-options load_from='yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'
```

3. 冻结 backbone 进行训练

通过 config 文件或者命令行中设置 model.backbone.frozen_stages=4 冻结 backbone 的 4 个 stages。

```shell
# 命令行中设置 model.backbone.frozen_stages=4
cd mmyolo
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py \
                      --cfg-options load_from='yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth' model.backbone.frozen_stages=4
```

### 训练验证中可视化相关

#### 验证阶段可视化

我们将 `configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py` 中的 `default_hooks` 的 `visualization` 进行修改，设置 `draw` 为 `True`，`interval` 为 `2`。

```shell
default_hooks = dict(
    logger=dict(interval=1),
    visualization=dict(draw=True, interval=2),
)
```

重新运行以下训练命令，在验证评估的过程中，每 `interval` 张图片就会保存一张标注结果和预测结果的拼图到 `work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon/{timestamp}/vis_data/vis_image` 文件夹中了。

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py
```

<div align=center>
<img src="https://moonstarimg.oss-cn-hangzhou.aliyuncs.com/img/20220920094007.png" alt="image"/>
</div>

#### 可视化后端使用

MMEngine 支持本地、TensorBoard 以及 wandb 等多种后端。

##### wandb 可视化使用

wandb 官网注册并在 https://wandb.ai/settings 获取到 wandb 的 API Keys。

<div align=center>
<img src="https://cdn.vansin.top/img/20220913212628.png" alt="image"/>
</div>

```shell
pip install wandb
# 运行了 wandb login 后输入上文中获取到的 API Keys ，便登录成功。
wandb login
```

在 `configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py` 添加 wandb 配置

```python
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
```

重新运行训练命令便可以在命令行中提示的网页链接中看到 loss、学习率和 coco/bbox_mAP 等数据可视化了。

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py
```

<div align=center>
<img src="https://cdn.vansin.top/img/20220913213221.png" alt="image"/>
</div>

##### Tensorboard 可视化使用

安装 Tensorboard 环境

```shell
pip install tensorboard
```

同上述在配置文件 `configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py`中添加 `tensorboard` 配置

```python
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
```

重新运行训练命令后，Tensorboard 文件会生成在可视化文件夹 `work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon/{timestamp}/vis_data` 下，
运行下面的命令便可以在网页链接使用 Tensorboard 查看 loss、学习率和 coco/bbox_mAP 等可视化数据了：

```shell
tensorboard --logdir=work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon
```

## 模型测试

```shell
python tools/test.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py \
                     work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon/epoch_300.pth \
                     --show-dir show_results
```

运行以上测试命令，推理结果图片会自动保存至 `work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon/{timestamp}/show_results` 文件夹中。下面为其中一张结果图片，左图为实际标注，右图为模型推理结果。

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/190913272-f99709e5-c798-46b8-aede-30f4e91683a3.jpg" alt="result_img"/>
</div>

## 模型部署

请参考[这里](../deploy/yolov5_deployment.md)
