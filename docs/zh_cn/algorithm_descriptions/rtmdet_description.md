# RTMDet 原理和实现全解析

## BBox Coder

RTMDet 的 BBox Coder 采用的是 `mmdet.DistancePointBBoxCoder`。

该类的 docstring 是这样的：

> This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left, right) and decode it back to the original.
>
> 这个编码器将 gt bboxes (x1, y1, x2, y2) 编码为 (top, bottom, left, right)，并且解码至原图像上

MMDet 编码的核心源码：

```python
def bbox2distance(points: Tensor, bbox: Tensor, max_dis: Optional[float] = None, eps: float = 0.1) -> Tensor:
    """
        points (Tensor): Shape (n, 2) or (b, n, 2), [x, y].
        bbox (Tensor): Shape (n, 4) or (b, n, 4), "xyxy" format
        max_dis (float, optional): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=
    """
    left = points[..., 0] - bbox[..., 0]
    top = points[..., 1] - bbox[..., 1]
    right = bbox[..., 2] - points[..., 0]
    bottom = bbox[..., 3] - points[..., 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)
```

MMDet 解码的核心源码：

```python
def distance2bbox(points: Tensor, distance: Tensor, max_shape: Optional[Union[Sequence[int], Tensor, Sequence[Sequence[int]]]] = None) -> Tensor:
    """
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Union[Sequence[int], Tensor, Sequence[Sequence[int]]],
            optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes

        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            ...
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes
```

## Loss

参与 Loss 计算的共有两个值：`loss_cls` 和 `loss_bbox`，其各自使用的 Loss 方法如下：

- `loss_cls`：`mmdet.QualityFocalLoss`
- `loss_bbox`：`mmdet.GIoULoss`

### QualityFocalLoss

Quality Focal Loss (QFL) 是 [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388) 的一部分。

普通的 Focal Loss 公式：

```{math}
\bold{FL}(p) = -(1-p_t)^\gamma\log(p_t),p_t = \begin{cases}
p, & \bold{when} \ y = 1 \\
1 - p, & \bold{when} \ y = 0
\end{cases}
\tag{1}
```

其中 $y\in\{1,0\}$ 指定真实类，$p\in[0,1]$ 表示标签 $y = 1$ 的类估计概率。$\gamma$ 是可调聚焦参数。具体来说，FL 由标准交叉熵部分 $-\log(p_t)$ 和动态比例因子部分 $-(1-p_t)^\gamma$ 组成，其中比例因子 $-(1-p_t)^\gamma$ 在训练期间自动降低简单类对于loss的比重，并且迅速将模型集中在困难类上。

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
        pred (torch.Tensor): Predicted joint representation of classification and quality (IoU) estimation with shape (N, C), C is the number of classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,) and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor. Defaults to 2.0.

    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label表示类别id，score表示质量分数
    label, score = target

    # 负样本由质量分数 0 来监督
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # 正样本有 bbox质量分数（IoU）来监督
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

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
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # 判断 Bbox 是空的或者 Bbox 最后一个尺寸的长度是 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # 批次尺寸必须相同
    # Batch dim 批次尺寸: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            ...

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        ...

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # 计算 gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss
```
