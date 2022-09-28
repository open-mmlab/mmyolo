# RTMDet 原理和实现全解析

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
        points (Tensor): 需要计算的点（该点已经乘上了 strides），Shape (n, 2) or (b, n, 2), [x, y].
        bbox (Tensor): BBox 的 xyxy，Shape (n, 4) or (b, n, 4), "xyxy" format.
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
        points (Tensor): 需要计算的点，Shape (B, N, 2) or (N, 2).
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
