# 可视化

## 特征图可视化

<div align=center>
<img src="https://user-images.githubusercontent.com/89863442/190903635-27bbc619-9bf8-43a8-aea8-ea13b9dad28c.jpg" width="1000"/>
</div>
可视化可以给深度学习的模型训练和测试过程提供直观解释。

MMYOLO 中，将使用 MMEngine 提供的 `Visualizer` 可视化器进行特征图可视化，其具备如下功能：

- 支持基础绘图接口以及特征图可视化
- 支持选择模型中的不同层来得到特征图，包含 `squeeze_mean` ， `select_max` ， `topk` 三种显示方式，用户还可以使用 `arrangement` 自定义特征图显示的布局方式。

## 特征图绘制

你可以调用 `demo/featmap_vis_demo.py` 来简单快捷地得到可视化结果，为了方便理解，将其主要参数的功能梳理如下：

- `img` 选择要用于特征图可视化的图片，支持单张图片或者图片路径列表

- `config` 选择算法的配置文件

- `checkpoint` 选择对应算法的权重文件

- `--out-file` 将得到的特征图保存到本地，并指定路径和文件名

- `--device` 指定用于推理图片的硬件，`--device cuda：0`  表示使用第 1 张 GPU 推理，`--device cpu` 表示用 CPU 推理

- `--score-thr` 设置检测框的置信度阈值，只有置信度高于这个值的框才会显示

- `--preview-model` 可以预览模型，方便用户理解模型的特征层结构

- `--target-layers` 对指定层获取可视化的特征图

  - 可以单独输出某个层的特征图，例如： `--target-layers backbone` ,  `--target-layers neck` ,  `--target-layers backbone.stage4` 等
  - 参数为列表时，也可以同时输出多个层的特征图，例如： `--target-layers backbone.stage4 neck` 表示同时输出 backbone 的 stage4 层和 neck 的三层一共四层特征图

- `--channel-reduction` 输入的 Tensor 一般是包括多个通道的，channel_reduction 参数可以将多个通道压缩为单通道，然后和图片进行叠加显示，有以下三个参数可以设置

  - `squeeze_mean` 将输入的 C 维度采用 mean 函数压缩为一个通道，输出维度变成 (1, H, W)
  - `select_max` 从输入的 C 维度中先在空间维度 sum，维度变成 (C, )，然后选择值最大的通道
  - `None` 表示不需要压缩，此时可以通过 topk 参数可选择激活度最高的 topk 个特征图显示

- `--topk` 只有在 `channel_reduction` 参数为 None 的情况下， topk 参数才会生效，其会按照激活度排序选择 topk 个通道，然后和图片进行叠加显示，并且此时会通过 `--arrangement` 参数指定显示的布局，该参数表示为一个数组，两个数字需要以空格分开，例如： `--topk 5 --arrangement 2 3` 表示以2行3列显示激活度排序最高的5张特征图， `--topk 7 --arrangement 3 3` 表示以3行3列显示激活度排序最高的7张特征图

  - 如果 topk 不是 -1，则会按照激活度排序选择 topk 个通道显示
  - 如果 topk = -1，此时通道 C 必须是 1 或者 3 表示输入数据是图片，否则报错提示用户应该设置 `channel_reduction` 来压缩通道。

- 考虑到输入的特征图通常非常小，函数默认将特征图进行上采样后方便进行可视化。

## 用法示例

以预训练好的 YOLOv5_s 模型为例:

请提前下载 YOLOv5_s 模型权重到本仓库根路径下：

```shell
cd mmyolo
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth
```

(1) 将多通道特征图采用 `select_max` 参数压缩为单通道并显示, 通过提取 backbone 层输出进行特征图可视化，将得到 backbone 三个输出层的特征图

```python
python demo/featmap_vis_demo.py demo/dog.jpg configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --target-layers backbone --channel-reduction select_max
```

<div align=center>
<img src="https://user-images.githubusercontent.com/89863442/190939711-c498060d-ed98-4dca-aa02-b41ca82f51b0.jpg" width="800"/>
</div>

(2) 将多通道特征图采用 `squeeze_mean` 参数压缩为单通道并显示, 通过提取 neck 层输出进行特征图可视化，将得到 neck 三个输出层的特征图

```python
python demo/featmap_vis_demo.py demo/dog.jpg configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --target-layers neck --channel-reduction squeeze_mean
```

<div align=center>
<img src="https://user-images.githubusercontent.com/89863442/190939718-b320239e-87f0-4a07-8525-b4087372e3fd.jpg" width="800"/>
</div>

(3) 将多通道特征图采用 `squeeze_mean` 参数压缩为单通道并显示, 通过提取 backbone.stage4 和 backbone.stage3 层输出进行特征图可视化，将得到两个输出层的特征图

```python
python demo/featmap_vis_demo.py demo/dog.jpg configs/yolov5/yolov5_s-v61_fast_syncbn_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --target-layers backbone.stage4 backbone.stage3 --channel-reduction squeeze_mean
```

<div align=center>
<img src="https://user-images.githubusercontent.com/89863442/190939720-cf829954-cc15-4b31-9be3-229d4d95e458.jpg" width="800"/>
</div>

(4) 利用 `--topk 3 --arrangement 2 2` 参数选择多通道特征图中激活度最高的 3 个通道并采用 2x2 布局显示, 用户可以通过 `arrangement` 参数选择自己想要的布局，特征图将自动布局，先按每个层中的 top3 特征图按 2x2 的格式布局，再将每个层按 2x2 布局

```python
python demo/featmap_vis_demo.py demo/dog.jpg configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --target-layers backbone.stage3 backbone.stage4 --channel-reduction None --topk 3 --arrangement 2 2 --out-file 4.jpg
```

<div align=center>
<img src="https://user-images.githubusercontent.com/89863442/190939723-911c5e9b-dd33-42eb-be4a-ba45f03110a0.jpg" width="1200"/>
</div>

(5) 存储绘制后的图片，在绘制完成后，可以选择本地窗口显示，也可以存储到本地，只需要加入参数 `--out-file xxx.jpg`

```python
python demo/featmap_vis_demo.py demo/dog.jpg configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --target-layers backbone --channel-reduction select_max --out-file featmap_backbone
```
