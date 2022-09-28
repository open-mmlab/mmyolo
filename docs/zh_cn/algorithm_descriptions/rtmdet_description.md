### 标签匹配策略

标签匹配策略 `Label Assignment` 是目标检测模型训练中最核心的问题之一,
更好的标签匹配策略往往能够使得网络更好学习到物体的特征以提高检测能力。

早期的样本标签匹配策略一般都是基于 `空间以及尺度信息的先验` 来决定样本的选取。
如 `FCOS` 中先限定网格中心点在 `GT` 内筛选后然后再通过不同特征层限制尺寸来决定正负样本;
`RetinaNet` 则是通过 `Anchor` 与 `GT` 的最大 `IOU` 匹配来划分正负样本;
`YOLOV5` 的正负样本则是通过样本的宽高比先筛选一部分, 然后通过位置信息选取
`GT` 中心落在的 `Grid` 以及临近的两个作为正样本。

但是上述方法都是属于基于 `先验` 的静态匹配策略, 就是样本的选取方式是根据人的经验规定的。
不会随着网络的优化而进行自动优化选取到更好的样本, 近些年涌现了许多优秀的动态标签匹配策略：
如 `OTA` 提出使用 `Sinkhorn` 迭代求解匹配中的最优传输问题, `YOLOX` 中使用 `OTA` 的近似算法
`SimOTA` , `TOOD` 将分类分数以及 `IOU` 相乘计算 `Cost` 矩阵进行标签匹配等等。
这些算法将 `预测的 Bboxes 与 GT 的 IOU ` 和 `分类分数`
或者是对应 `分类Loss` 和 `回归Loss` 拿来计算 `Matching Cost` 矩阵再通过 `top-k`
的方式动态决定样本选取以及样本个数。通过这种方式,
在网络优化的过程中会自动选取对分类或者回归更加敏感有效的位置的样本,
它不再只依赖先验的静态的信息, 而是使用当前的预测结果去动态寻找最优的匹配,
只要模型的预测越准确, 匹配算法求得的结果也会更优秀。但是在网络训练的初期,
网络的分类以及回归是随机初始化, 这个时候还是需要 `先验` 来约束, 以达到 `冷启动` 的效果。

`RTMDet` 作者也是采用了动态的 `SimOTA` 做法，不过其对动态的正负样本分配策略进行了改进。
之前的动态匹配策略（ `HungarianAssigner` 、`OTA` ）往往使用与 `Loss`
完全一致的代价函数作为匹配的依据，但我们经过实验发现这并不一定时最优的。
使用更多 `Soften` 的 `Cost` 以及先验，能够提升性能。

综上, `RTMDet` 提出了 `Dynamic Soft Label Assigner` 来实现标签的动态匹配策略,
该方法主要包括使用
**位置先验信息损失** , **样本回归损失** , **样本分类损失** , 同时对三个损失进行了 `Soft`
处理进行参数调优, 以达到最佳的动态匹配效果。

该方法 Matching Cost 矩阵由如下损失构成：

```python
cost_matrix = soft_cls_cost + iou_cost + soft_center_prior
```

1. Soft_Center_Prior

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

```python
# 计算回归 bboxes 和 gts 的 iou
pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
# 将 iou 使用 log 进行 soft , iou 越小 cost 更小
iou_cost = -torch.log(pairwise_ious + EPS) * 3
```

3. Soft_Cls_Cost

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

通过计算上述三个损失的和得到最终的 `cost_matrix` 后, 再使用 `SimOTA` 决定每一个 `GT` 匹配的样本的个数并决定最终的样本
具体操作如下所示：

1. 首先通过自适应计算每一个 `gt` 要选取的样本数量：
   取每一个 `gt` 与所有 `bboxes` 前 `13` 大的 `iou`,
   得到它们的和取整后作为这个 `gt` 的 `样本数目` , 最少为 `1` 个,
   记为 `dynamic_ks`。
2. 对于每一个 `gt` , 将其 `cost_matrix` 矩阵前 `dynamic_ks` 小的位置作为该 `gt` 的正样本。
3. 对于某一个 `bbox`, 如果被匹配到多个 `gt`
   就将与这些 `gts` 的 `cost_marix` 中最小的那个作为其 `label`。

在网络训练初期，因参数初始化，回归和分类的损失值 `Cost` 往往较大, 这时候 `IOU` 比较小，
选取的样本较少，主要起作用的是 `Soft_center_prior` 也就是位置信息，优先选取位置距离 `GT`
比较近的样本作为正样本，这也符合人们的理解，在网络前期给少量并且有足够质量的样本，以达到冷启动。
当网络进行训练一段时间过后，分类分支和回归分支都进行了一定的优化后，这时 `IOU` 变大，
选取的样本也逐渐增多，这时网络也有能力学习到更多的样本，同时因为 `IOU_Cost` 以及 `Soft_Cls_Cost`
变小，网络也会动态的找到更有利优化分类以及回归的样本点。

在 `Resnet50-1x` 的三种损失的消融实验：

| Soft_cls_cost | Soft_center_prior | Log_IoU_cost | mAP  |
| :-----------: | :---------------: | :----------: | :--: |
|       ×       |         ×         |      ×       | 39.9 |
|       √       |         ×         |      ×       | 40.3 |
|       √       |         √         |      ×       | 40.8 |
|       √       |         √         |      √       | 41.3 |

与其他主流 `Assign` 方法在 `Resnet50-1x` 的对比实验：

|    method     | mAP  |
| :-----------: | :--: |
|     ATSS      | 39.2 |
|      PAA      | 40.4 |
|      OTA      | 40.7 |
| TOOD(w/o TAH) | 40.7 |
|     Ours      | 41.3 |

无论是 `Resnet50-1x` 还是标准的设置下，还是在`300epoch` + `havy augmentation`,
相比于 `SimOTA` 、 `OTA` 以及 `TOOD` 中的 `TAL` 均有提升。

| 300e + Mosaic & MixUP | mAP  |
| :-------------------- | :--- |
| RTMDet-s + SimOTA     | 43.2 |
| RTMDet-s + DSLA       | 44.5 |
