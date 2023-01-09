# Conventions

Please check the following conventions if you would like to modify MMYOLO as your own project.

## About the order of image shape

In OpenMMLab 2.0, to be consistent with the input argument of OpenCV, the argument about image shape in the data transformation pipeline is always in the `(width, height)` order. On the contrary, for computation convenience, the order of the field going through the data pipeline and the model is `(height, width)`. Specifically, in the results processed by each data transform pipeline, the fields and their value meaning is as below:

- img_shape: (height, width)
- ori_shape: (height, width)
- pad_shape: (height, width)
- batch_input_shape: (height, width)

As an example, the initialization arguments of `Mosaic` are as below:

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

       # img_scale order should be (width, height)
       self.img_scale = img_scale

    def transform(self, results: dict) -> dict:
        ...

        results['img'] = mosaic_img
        # (height, width)
        results['img_shape'] = mosaic_img.shape[:2]
```
