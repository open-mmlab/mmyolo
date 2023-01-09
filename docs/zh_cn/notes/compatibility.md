# MMYOLO 兼容性说明

## MMYOLO v0.3.0

### METAINFO 修改

为了和 OpenMMLab 其他仓库统一，将 Dataset 里 `METAINFO` 的所有键从大写改为小写。

| 在 v0.3.0 之前 | v0.3.0 及之后 |
| :------------: | :-----------: |
|    CLASSES     |    classes    |
|    PALETTE     |    palette    |
|  DATASET_TYPE  | dataset_type  |

### 关于图片 shape 顺序的说明

在 OpenMMLab 2.0 中， 为了与 OpenCV 的输入参数相一致，图片处理 pipeline 中关于图像 shape 的输入参数总是以 `(width, height)` 的顺序排列。
相反，为了计算方便，经过 pipeline 和 model 的字段的顺序是 `(height, width)`。具体来说在每个数据 pipeline 处理的结果中，字段和它们的值含义如下：

- img_shape: (height, width)
- ori_shape: (height, width)
- pad_shape: (height, width)
- batch_input_shape: (height, width)

以 `Mosaic` 为例，其初始化参数如下所示：

```python
@TRANSFORMS.register_module()
class Mosaic(BaseTransform):
    def __init__(self,
                img_scale: Tuple[int, int] = (640, 640),
                center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                bbox_clip_border: bool = True,
                pad_val: float = 114.0,
                prob: float = 1.0) -> None:
       ...

       # img_scale 顺序应该是 (width, height)
       self.img_scale = img_scale

    def transform(self, results: dict) -> dict:
        ...

        results['img'] = mosaic_img
        # (height, width)
        results['img_shape'] = mosaic_img.shape[:2]
```
