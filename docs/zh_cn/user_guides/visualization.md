# 可视化

本文包括特征图可视化和 Grad-Based 和 Grad-free CAM 可视化

## 特征图可视化

<div align=center>
<img src="https://user-images.githubusercontent.com/89863442/190903635-27bbc619-9bf8-43a8-aea8-ea13b9dad28c.jpg" width="1000" alt="image"/>
</div>
可视化可以给深度学习的模型训练和测试过程提供直观解释。

MMYOLO 中，将使用 MMEngine 提供的 `Visualizer` 可视化器进行特征图可视化，其具备如下功能：

- 支持基础绘图接口以及特征图可视化。
- 支持选择模型中的不同层来得到特征图，包含 `squeeze_mean` ， `select_max` ， `topk` 三种显示方式，用户还可以使用 `arrangement` 自定义特征图显示的布局方式。

### 特征图绘制

你可以调用 `demo/featmap_vis_demo.py` 来简单快捷地得到可视化结果，为了方便理解，将其主要参数的功能梳理如下：

- `img`：选择要用于特征图可视化的图片，支持单张图片或者图片路径列表。

- `config`：选择算法的配置文件。

- `checkpoint`：选择对应算法的权重文件。

- `--out-file`：将得到的特征图保存到本地，并指定路径和文件名。

- `--device`：指定用于推理图片的硬件，`--device cuda：0`  表示使用第 1 张 GPU 推理，`--device cpu` 表示用 CPU 推理。

- `--score-thr`：设置检测框的置信度阈值，只有置信度高于这个值的框才会显示。

- `--preview-model`：可以预览模型，方便用户理解模型的特征层结构。

- `--target-layers`：对指定层获取可视化的特征图。

  - 可以单独输出某个层的特征图，例如： `--target-layers backbone` ,  `--target-layers neck` ,  `--target-layers backbone.stage4` 等。
  - 参数为列表时，也可以同时输出多个层的特征图，例如： `--target-layers backbone.stage4 neck` 表示同时输出 backbone 的 stage4 层和 neck 的三层一共四层特征图。

- `--channel-reduction`：输入的 Tensor 一般是包括多个通道的，`channel_reduction` 参数可以将多个通道压缩为单通道，然后和图片进行叠加显示，有以下三个参数可以设置：

  - `squeeze_mean`：将输入的 C 维度采用 mean 函数压缩为一个通道，输出维度变成 (1, H, W)。
  - `select_max`：将输入先在空间维度 sum，维度变成 (C, )，然后选择值最大的通道。
  - `None`：表示不需要压缩，此时可以通过 `topk` 参数可选择激活度最高的 `topk` 个特征图显示。

- `--topk`：只有在 `channel_reduction` 参数为 `None` 的情况下， `topk` 参数才会生效，其会按照激活度排序选择 `topk` 个通道，然后和图片进行叠加显示，并且此时会通过 `--arrangement` 参数指定显示的布局，该参数表示为一个数组，两个数字需要以空格分开，例如： `--topk 5 --arrangement 2 3` 表示以 `2行 3列` 显示激活度排序最高的 5 张特征图， `--topk 7 --arrangement 3 3` 表示以 `3行 3列` 显示激活度排序最高的 7 张特征图。

  - 如果 topk 不是 -1，则会按照激活度排序选择 topk 个通道显示。
  - 如果 topk = -1，此时通道 C 必须是 1 或者 3 表示输入数据是图片，否则报错提示用户应该设置 `channel_reduction` 来压缩通道。

- 考虑到输入的特征图通常非常小，函数默认将特征图进行上采样后方便进行可视化。

**注意：当图片和特征图尺度不一样时候，`draw_featmap` 函数会自动进行上采样对齐。如果你的图片在推理过程中前处理存在类似 Pad 的操作此时得到的特征图也是 Pad 过的，那么直接上采样就可能会出现不对齐问题。**

### 用法示例

以预训练好的 YOLOv5-s 模型为例:

请提前下载 YOLOv5-s 模型权重到本仓库根路径下：

```shell
cd mmyolo
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth
```

(1) 将多通道特征图采用 `select_max` 参数压缩为单通道并显示, 通过提取 `backbone` 层输出进行特征图可视化，将得到 `backbone` 三个输出层的特征图：

```shell
python demo/featmap_vis_demo.py demo/dog.jpg \
                                configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                                yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
                                --target-layers backbone \
                                --channel-reduction select_max
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/198520580-c1b24d50-2e90-4ba5-af51-5a7dcb9db945.png" width="800" alt="image"/>
</div>

实际上上述代码存在图片和特征图不对齐问题，解决办法有两个：

1. 修改 YOLOv5 配置，让后处理只是简单的 Resize 即可，这对于可视化是没有啥影响的

2. 可视化时候图片应该用前处理后的，而不能用前处理前的

**为了简单目前这里采用第一种解决办法，后续会采用第二种方案修复，让大家可以不修改配置即可使用**。具体来说是将原先的 `test_pipeline` 替换为仅仅 Resize 版本。

旧的 `test_pipeline` 为：

```python
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
```

修改为如下配置：

```python
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=_base_.file_client_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False), # 这里将 LetterResize 修改成 mmdet.Resize
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
```

正确效果如下：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/198521116-dbccade4-370c-4051-92bf-923ca8f60f24.png" width="800" alt="image"/>
</div>

(2) 将多通道特征图采用 `squeeze_mean` 参数压缩为单通道并显示, 通过提取 `neck` 层输出进行特征图可视化，将得到 `neck` 三个输出层的特征图：

```shell
python demo/featmap_vis_demo.py demo/dog.jpg \
                                configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                                yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
                                --target-layers neck \
                                --channel-reduction squeeze_mean
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/198521267-20202e3d-b1bc-4559-9085-e0af287636c8.png" width="800" alt="image"/>
</div>

(3) 将多通道特征图采用 `squeeze_mean` 参数压缩为单通道并显示, 通过提取 `backbone.stage4` 和 `backbone.stage3` 层输出进行特征图可视化，将得到两个输出层的特征图：

```shell
python demo/featmap_vis_demo.py demo/dog.jpg \
                                configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                                yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
                                --target-layers backbone.stage4 backbone.stage3 \
                                --channel-reduction squeeze_mean
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/198522004-c5782807-166a-45f3-96e3-7e6df5dc70ac.png" width="800" alt="image"/>
</div>

(4) 利用 `--topk 3 --arrangement 2 2` 参数选择多通道特征图中激活度最高的 3 个通道并采用 `2x2` 布局显示, 用户可以通过 `arrangement` 参数选择自己想要的布局，特征图将自动布局，先按每个层中的 `top3` 特征图按 `2x2` 的格式布局，再将每个层按 `2x2` 布局：

```shell
python demo/featmap_vis_demo.py demo/dog.jpg \
                                configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                                yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
                                --target-layers backbone.stage3 backbone.stage4 \
                                --channel-reduction None \
                                --topk 3 \
                                --arrangement 2 2
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/198522489-8adee6ae-9915-4e9d-bf50-167b8a12c275.png" width="800" alt="image"/>
</div>

(5) 存储绘制后的图片，在绘制完成后，可以选择本地窗口显示，也可以存储到本地，只需要加入参数 `--out-file xxx.jpg`：

```shell
python demo/featmap_vis_demo.py demo/dog.jpg \
                                configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                                yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
                                --target-layers backbone \
                                --channel-reduction select_max \
                                --out-file featmap_backbone.jpg
```

## Grad-Based 和 Grad-free CAM 可视化

目标检测 CAM 可视化相比于分类 CAM 复杂很多且差异很大。本文只是简要说明用法，后续会单独开文档详细描述实现原理和注意事项。

你可以调用 `demo/boxmap_vis_demo.py` 来简单快捷地得到 Box 级别的 AM 可视化结果，目前已经支持 YOLOv5/YOLOv6/YOLOX/RTMDet。

以 YOLOv5 为例，和特征图可视化绘制一样，你需要先修改 test_pipeline，否则会出现特征图和原图不对齐问题。

旧的 `test_pipeline` 为：

```python
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
```

修改为如下配置：

```python
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=_base_.file_client_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=False), # 这里将 LetterResize 修改成 mmdet.Resize
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
```

(1) 使用 gradcam 方法可视化 neck 模块的最后一个输出层的 am 图

```shell
python demo/boxam_vis_demo.py \
        demo/dog.jpg \
        configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
        yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth

```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/203775584-c4aebf11-4ff8-4530-85fe-7dda897e95a8.jpg" width="800" alt="image"/>
</div>

相对应的特征图 am 图如下：

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/203774801-1555bcfb-a8f9-4688-8ed6-982d6ad38e1d.jpg" width="800" alt="image"/>
</div>

可以看出 gradcam 效果可以突出 box 级别的 am 信息。

你可以通过 --topk 参数选择仅仅可视化预测分值最好的前几个预测框

```shell
python demo/boxam_vis_demo.py \
        demo/dog.jpg \
        configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
        yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
        --topk 2
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/203778700-3165aa72-ecaf-40cc-b470-6911646e6046.jpg" width="800" alt="image"/>
</div>

(2) 使用 ablationcam 方法可视化 neck 模块的最后一个输出层的 am 图

```shell
python demo/boxam_vis_demo.py \
        demo/dog.jpg \
        configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
        yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
        --method ablationcam
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/203776978-b5a9b383-93b4-4b35-9e6a-7cac684b372c.jpg" width="800" alt="image"/>
</div>

由于 ablationcam 是通过每个通道对分值的贡献程度来加权，因此无法实现类似 gradcam 的仅仅可视化 box 级别的 am 信息。 但是你引入可以使用 --norm-in-bbox 来仅仅显示 bbox 内部 am

```shell
python demo/boxam_vis_demo.py \
        demo/dog.jpg \
        configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
        yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
        --method ablationcam \
        --norm-in-bbox
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/203777566-7c74e82f-b477-488e-958f-91e1d10833b9.jpg" width="800" alt="image"/>
</div>
