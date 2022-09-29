# RTMDet 原理和实现全解析

## 0 简介

### 1.3 正负样本匹配策略

正负样本匹配策略或者称为标签匹配策略 `Label Assignment` 是目标检测模型训练中最核心的问题之一, 更好的标签匹配策略往往能够使得网络更好学习到物体的特征以提高检测能力。

早期的样本标签匹配策略一般都是基于 `空间以及尺度信息的先验` 来决定样本的选取。 典型案例如下：

- `FCOS` 中先限定网格中心点在 `GT` 内筛选后然后再通过不同特征层限制尺寸来决定正负样本
- `RetinaNet` 则是通过 `Anchor` 与 `GT` 的最大 `IOU` 匹配来划分正负样本
- `YOLOV5` 的正负样本则是通过样本的宽高比先筛选一部分, 然后通过位置信息选取 `GT` 中心落在的 `Grid` 以及临近的两个作为正样本

但是上述方法都是属于基于 `先验` 的静态匹配策略, 就是样本的选取方式是根据人的经验规定的。 不会随着网络的优化而进行自动优化选取到更好的样本, 近些年涌现了许多优秀的动态标签匹配策略：

- `OTA` 提出使用 `Sinkhorn` 迭代求解匹配中的最优传输问题
- `YOLOX` 中使用 `OTA` 的近似算法 `SimOTA` , `TOOD` 将分类分数以及 `IOU` 相乘计算 `Cost` 矩阵进行标签匹配等等

这些算法将 `预测的 Bboxes 与 GT 的 IOU ` 和 `分类分数` 或者是对应 `分类 Loss` 和 `回归 Loss` 拿来计算 `Matching Cost` 矩阵再通过 `top-k` 的方式动态决定样本选取以及样本个数。通过这种方式,
在网络优化的过程中会自动选取对分类或者回归更加敏感有效的位置的样本, 它不再只依赖先验的静态的信息, 而是使用当前的预测结果去动态寻找最优的匹配, 只要模型的预测越准确, 匹配算法求得的结果也会更优秀。但是在网络训练的初期, 网络的分类以及回归是随机初始化, 这个时候还是需要 `先验` 来约束, 以达到 `冷启动` 的效果。

`RTMDet` 作者也是采用了动态的 `SimOTA` 做法，不过其对动态的正负样本分配策略进行了改进。 之前的动态匹配策略（ `HungarianAssigner` 、`OTA` ）往往使用与 `Loss` 完全一致的代价函数作为匹配的依据，但我们经过实验发现这并不一定时最优的。 使用更多 `Soften` 的 `Cost` 以及先验，能够提升性能。

综上, `RTMDet` 提出了 `Dynamic Soft Label Assigner` 来实现标签的动态匹配策略, 该方法主要包括使用 **位置先验信息损失** , **样本回归损失** , **样本分类损失** , 同时对三个损失进行了 `Soft` 处理进行参数调优, 以达到最佳的动态匹配效果。

该方法 Matching Cost 矩阵由如下损失构成：

```python
cost_matrix = soft_cls_cost + iou_cost + soft_center_prior
```

1. Soft_Center_Prior

$$
C\_{center} = \\alpha^{|x\_{pred}-x\_{gt}|-\\beta}
$$

```python
# valid_prior Tensor[N,4] 表示anchor point
# 4分别表示 x, y, 以及对应的特征层的 stride, stride
gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
valid_prior = priors[valid_mask]
strides = valid_prior[:, 2]
# 计算gt与anchor point的中心距离并转换到特征图尺度
distance = (valid_prior[:, None, :2] - gt_center[None, :, :]
                    ).pow(2).sum(-1).sqrt() / strides[:, None]
# 以10为底计算位置的软化损失,限定在gt的6个单元格以内
soft_center_prior = torch.pow(10, distance - 3)
```

2. IOU_Cost

$$
C\_{reg} = -log(IOU)
$$

```python
# 计算回归 bboxes 和 gts 的 iou
pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
# 将 iou 使用 log 进行 soft , iou 越小 cost 更小
iou_cost = -torch.log(pairwise_ious + EPS) * 3
```

3. Soft_Cls_Cost

$$
C\_{cls} = CE(P,Y\_{soft}) \*(Y\_{soft}-P)^2
$$

```python
# 生成分类标签
 gt_onehot_label = (
    F.one_hot(gt_labels.to(torch.int64),
              pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                  num_valid, 1, 1))
valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
# 不单单将分类标签为01,而是换成与 gt 的 iou
soft_label = gt_onehot_label * pairwise_ious[..., None]
# 使用 quality focal loss 计算分类损失 cost ,与实际的分类损失计算保持一致
scale_factor = soft_label - valid_pred_scores.sigmoid()
soft_cls_cost = F.binary_cross_entropy_with_logits(
    valid_pred_scores, soft_label,
    reduction='none') * scale_factor.abs().pow(2.0)
soft_cls_cost = soft_cls_cost.sum(dim=-1)
```

通过计算上述三个损失的和得到最终的 `cost_matrix` 后, 再使用 `SimOTA` 决定每一个 `GT` 匹配的样本的个数并决定最终的样本。具体操作如下所示：

1. 首先通过自适应计算每一个 `gt` 要选取的样本数量： 取每一个 `gt` 与所有 `bboxes` 前 `13` 大的 `iou`, 得到它们的和取整后作为这个 `gt` 的 `样本数目` , 最少为 `1` 个, 记为 `dynamic_ks`。
2. 对于每一个 `gt` , 将其 `cost_matrix` 矩阵前 `dynamic_ks` 小的位置作为该 `gt` 的正样本。
3. 对于某一个 `bbox`, 如果被匹配到多个 `gt` 就将与这些 `gts` 的 `cost_marix` 中最小的那个作为其 `label`。

在网络训练初期，因参数初始化，回归和分类的损失值 `Cost` 往往较大, 这时候 `IOU` 比较小， 选取的样本较少，主要起作用的是 `Soft_center_prior` 也就是位置信息，优先选取位置距离 `GT` 比较近的样本作为正样本，这也符合人们的理解，在网络前期给少量并且有足够质量的样本，以达到冷启动。
当网络进行训练一段时间过后，分类分支和回归分支都进行了一定的优化后，这时 `IOU` 变大， 选取的样本也逐渐增多，这时网络也有能力学习到更多的样本，同时因为 `IOU_Cost` 以及 `Soft_Cls_Cost` 变小，网络也会动态的找到更有利优化分类以及回归的样本点。

在 `Resnet50-1x` 的三种损失的消融实验：

| Soft_cls_cost | Soft_center_prior | Log_IoU_cost | mAP  |
| :------------ | :---------------- | :----------- | :--- |
| ×             | ×                 | ×            | 39.9 |
| √             | ×                 | ×            | 40.3 |
| √             | √                 | ×            | 40.8 |
| √             | √                 | √            | 41.3 |

与其他主流 `Assign` 方法在 `Resnet50-1x` 的对比实验：

|    method     | mAP  |
| :-----------: | :--- |
|     ATSS      | 39.2 |
|      PAA      | 40.4 |
|      OTA      | 40.7 |
| TOOD(w/o TAH) | 40.7 |
|     Ours      | 41.3 |

无论是 `Resnet50-1x` 还是标准的设置下，还是在`300epoch` + `havy augmentation`,  相比于 `SimOTA` 、 `OTA` 以及 `TOOD` 中的 `TAL` 均有提升。

| 300e + Mosaic & MixUP | mAP  |
| :-------------------- | :--- |
| RTMDet-s + SimOTA     | 43.2 |
| RTMDet-s + DSLA       | 44.5 |

## BBox Coder

RTMDet 的 BBox Coder 采用的是 `mmdet.DistancePointBBoxCoder`。

该类的 docstring 是这样的：

> This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left, right) and decode it back to the original.
>
> 这个编码器将 gt bboxes (x1, y1, x2, y2) 编码为 (top, bottom, left, right)，并且解码至原图像上

MMDet 编码的核心源码：

```python
def bbox2distance(points: Tensor, bbox: Tensor, ...) -> Tensor:
    """
        points (Tensor): 相当于 scale 值 stride ，且每个预测点仅为一个正方形 anchor 的 anchor point [x, y]，Shape (n, 2) or (b, n, 2).
        bbox (Tensor): Bbox 为乘上 stride 的网络预测值，格式为 xyxy，Shape (n, 4) or (b, n, 4).
    """
    # 计算点距离四边的距离
    left = points[..., 0] - bbox[..., 0]
    top = points[..., 1] - bbox[..., 1]
    right = bbox[..., 2] - points[..., 0]
    bottom = bbox[..., 3] - points[..., 1]

    ...

    return torch.stack([left, top, right, bottom], -1)
```

MMDet 解码的核心源码：

```python
def distance2bbox(points: Tensor, distance: Tensor, ...) -> Tensor:
    """
        通过距离反算 bbox 的 xyxy
        points (Tensor): 正方形的预测 anchor 的 anchor point [x, y]，Shape (B, N, 2) or (N, 2).
        distance (Tensor): 距离四边的距离。(left, top, right, bottom). Shape (B, N, 4) or (N, 4)
    """

    # 反算 bbox xyxy
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    ...

    return bboxes
```

## Loss

参与 Loss 计算的共有两个值：`loss_cls` 和 `loss_bbox`，其各自使用的 Loss 方法如下：

- `loss_cls`：`mmdet.QualityFocalLoss`
- `loss_bbox`：`mmdet.GIoULoss`

### QualityFocalLoss

Quality Focal Loss (QFL) 是 [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388) 的一部分。

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/192767279-4e69f935-1685-4089-82a3-0add201f98cc.png" alt="image"/>
</div>

普通的 Focal Loss 公式：

```{math}
\bold{FL}(p) = -(1-p_t)^\gamma\log(p_t),p_t = \begin{cases}
p, & \bold{when} \ y = 1 \\
1 - p, & \bold{when} \ y = 0
\end{cases}
\tag{1}
```

其中 $y\in\{1,0\}$ 指定真实类，$p\in[0,1]$ 表示标签 $y = 1$ 的类估计概率。$\gamma$ 是可调聚焦参数。具体来说，FL 由标准交叉熵部分 $-\log(p_t)$ 和动态比例因子部分 $-(1-p_t)^\gamma$ 组成，其中比例因子 $-(1-p_t)^\gamma$ 在训练期间自动降低简单类对于 loss 的比重，并且迅速将模型集中在困难类上。

首先 $y = 0$ 表示质量得分为 0 的负样本，$0 < y \leq1$ 表示目标 IoU 得分为 y 的正样本。为了针对连续的标签，扩展 FL 的两个部分：
1. 交叉熵部分 $-\log(p_t)$ 扩展为完整版本 $-((1-y)\log(1-\sigma)+y\log(\sigma))$;
2. 比例因子部分 $-(1-p_t)^\gamma$ 被泛化为估计 $\gamma$ 与其连续标签 $y$ 的绝对距离，即 $|y-\sigma|^\beta (\beta \geq 0)$。

结合上面两个部分之后，我们得出 QFL 的公式：

```{math}
\bold{QFL}(\sigma) = -|y-\sigma|^\beta((1-y)\log(1-\sigma)+y\log(\sigma))
```

具体作用是：可以将离散标签的 `focal loss` 泛化到连续标签上，将 bboxes 与 gt 的 IoU 的作为分类分数的标签，使得分类分数为表征回归质量的分数。

MMDet 实现源码的核心部分：

```python
@weighted_loss
def quality_focal_loss(pred, target, beta=2.0):
    """
        pred (torch.Tensor): 用形状（N，C）联合表示预测分类和质量（IoU），C是类的数量。
        target (tuple([torch.Tensor])): 目标类别标签的形状为（N，），目标质量标签的形状是（N，，）。
        beta (float): 计算比例因子的 β 参数.
    """
    ...

    # label表示类别id，score表示质量分数
    label, score = target

    # 负样本质量分数0来进行监督
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)

    # 计算交叉熵部分的值
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # 得出 IoU 在区间 (0,1] 的 bbox
    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()

    # 正样本由 IoU 范围在 (0,1] 的 bbox 来监督
    # 计算动态比例因子
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]

    # 计算两部分的 loss
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    # 得出最终 loss
    loss = loss.sum(dim=1, keepdim=False)
    return loss
```

### GIoULoss

论文：[Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630)

GIoU Loss 用于计算两个框重叠区域的关系，重叠区域越大，损失越小，反之越大。而且 GIoU 是在 \[0,2\] 之间，因为其值被限制在了一个较小的范围内，所以网络不会出现剧烈的波动，证明了其具有比较好的稳定性。

下图是基本的实现流程图：

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/192568784-3884b677-d8e1-439c-8bd2-20943fcedd93.png" alt="image"/>
</div>

MMDet 实现源码的核心部分：

```python
def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    ...

    # 求两个区域的面积
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        # 得出两个 bbox 重合的左上角 lt 和右下角 rb
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        # 求重合面积
        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            ...
        else:
            union = area1
        if mode == 'giou':
            # 得出两个 bbox 最小凸闭合框的左上角 lt 和右下角 rb
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        ...

    # 求重合面积 / gt bbox 面积 的比率，即 IoU
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    ...

    # 求最小凸闭合框面积
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)

    # 计算 giou
    gious = ious - (enclose_area - union) / enclose_area
    return gious

@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss
```
