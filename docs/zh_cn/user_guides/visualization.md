# MMYOLO特征图可视化
<div align=center>
<img src="https://github.com/yang-0201/zexi/blob/74566a4e9439be2aaafc1d2144e816c7b0908ba2/featmap.jpg" width="1000"/>
</div>
可视化可以给深度学习的模型训练和测试过程提供直观解释。

在MMYOLO中，将使用MMEngine 提供的 `Visualizer` 可视化器用以可视化和存储模型在不同结构位置的特征图，具备如下功能：

- 支持基础绘图接口以及特征图可视化
- 支持选择模型中的不同层来得到特征图，包含`squeeze_mean`，`select_max`，`topk`三种显示方式，用户还可以使用`arrangement`自定义特征图显示的布局方式。


## 特征图绘制

你可以调用`demo/featmap_vis_demo.py`来简单快捷地得到可视化结果，为了方便理解，将其主要参数的功能梳理如下：
- `img` 选择要用于特征图可视化的图片，目前只支持单张特征图的可视化

- `config` 选择算法的配置文件

- `checkpoint` 选择对应算法的权重文件

- `--out-file` 将得到的特征图保存到本地，并指定路径和文件名

- `--target-layers` 对指定层获取可视化的特征图，例如：`backbone`, `neck`, `backbone.stage4`等

- `--channel-reduction` 输入的 Tensor 一般是包括多个通道的，channel_reduction 参数可以将多个通道压缩为单通道，然后和图片进行叠加显示，有以下三个参数可以设置

  - `squeeze_mean` 将输入的 C 维度采用 mean 函数压缩为一个通道，输出维度变成 (1, H, W)
  - `select_max` 从输入的 C 维度中先在空间维度 sum，维度变成 (C, )，然后选择值最大的通道
  - `None` 表示不需要压缩，此时可以通过 topk 参数可选择激活度最高的 topk 个特征图显示

- `--topk` 只有在`channel_reduction`参数为 None 的情况下，topk 参数才会生效，其会按照激活度排序选择 topk 个通道，然后和图片进行叠加显示，并且此时会通过`--arrangement`参数指定显示的布局
，例如：`--topk`参数设为5，`--arrangement`参数设为（2，3） 表示以两行三列显示激活度排序最高的5张特征图

  - 如果 topk 不是 -1，则会按照激活度排序选择 topk 个通道显示
  - 如果 topk = -1，此时通道 C 必须是 1 或者 3 表示输入数据是图片，否则报错提示用户应该设置 `channel_reduction`来压缩通道。

- 考虑到输入的特征图通常非常小，函数默认将特征图进行上采样后方便进行可视化。

## 用法示例
以预训练好的yolov5_s模型为例:

(1) 将多通道特征图采用 `select_max` 参数压缩为单通道并显示, 通过提取backbone层输出进行特征图可视化，将得到backbone三个输出层的特征图

```python
python demo/featmap_vis_demo.py demo/dog.jpg configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py mmyolov5s.pt --target-layers backbone --channel-reduction select_max
```

<div align=center>
<img src="https://github.com/yang-0201/zexi/blob/13aea86be52aecb5b968e20cb8f6aae8294f8c0a/1.jpg" width="400"/>
</div>

(2) 将多通道特征图采用 `squeeze_mean` 参数压缩为单通道并显示, 通过提取neck层输出进行特征图可视化，将得到neck三个输出层的特征图

```python
python demo/featmap_vis_demo.py demo/dog.jpg configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py mmyolov5s.pt --target-layers neck --channel-reduction squeeze_mean
```

<div align=center>
<img src="https://github.com/yang-0201/zexi/blob/74566a4e9439be2aaafc1d2144e816c7b0908ba2/2.jpg" width="400"/>
</div>

(3) 将多通道特征图采用 `squeeze_mean` 参数压缩为单通道并显示, 通过提取backbone.stage4和backbone.stage3层输出进行特征图可视化，将得到两个输出层的特征图


```python
python demo/featmap_vis_demo.py demo/dog.jpg configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py mmyolov5s.pt --target-layers backbone.stage4 backbone.stage3 --channel-reduction squeeze_mean
```

<div align=center>
<img src="https://github.com/yang-0201/zexi/blob/74566a4e9439be2aaafc1d2144e816c7b0908ba2/3.jpg" width="400"/>
</div>

(4) 利用 `topk=5` 参数选择多通道特征图中激活度最高的 5 个通道并采用 2x3 布局显示, 用户可以通过 arrangement参数选择自己想要的布局

```python
python demo/featmap_vis_demo.py demo/dog.jpg configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py mmyolov5s.pt --target-layers neck --channel-reduction None --topk 5 arrangement (2,3)
```

<div align=center>
<img src="https://github.com/yang-0201/zexi/blob/74566a4e9439be2aaafc1d2144e816c7b0908ba2/4.jpg" width="1000"/>
</div>

## 保存特征图

在绘制完成后，可以选择本地窗口显示，也可以存储到本地

**(1) 存储绘制后的图片**

```python
python demo/featmap_vis_demo.py demo/dog.jpg configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py mmyolov5s.pt --target-layers backbone --channel-reduction select_max --out-file featmap_backbone
```
