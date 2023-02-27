# Automatic mixed precision（AMP）training

## AMP Operation Process

PyTorch's official AMP usage highlights automatic hybrid precision conversion, while MMCV introduces a relatively large number of manual conversion operations for compatibility between the two APIs. The core of mixed precision training is `with autocast`, where you don't need to think about the `fp16` setting in that context, but will automatically infer whether to use `fp16` or `fp32` based on the OP, while the MMCV implementation also calls the PyTorch implementation of AMP, but on PyTorch versions greater than or equal to 1.6 In the MMCV implementation, although it also calls the PyTorch implementation of AMP, in PyTorch versions greater than or equal to 1.6, it also additionally performs forced type conversions (why forced type conversions, because currently PyTorch AMP still has minor problems in some scenarios, and the program will report errors), which may be somewhat different from the official AMP behavior in some occasions.

1. for some OPs (`addcdiv, addcmul, atan2, bilinear, cat, cross, dot, equal, index_put, stack, tensordot`), such as `cat`, the widest range of types will prevail in PyTorch AMP, such as the `dot` operation , if `fp16` and `fp32` operate, then the output type is `fp32`, if they are both `fp16`, the `fp16` operation will be used, if they are both of type `fp32`, the output will still be `fp32`, but that `fp32` type will be forced to be the `fp16` operation in MMCV.
In short, the first two cases in the PyTorch AMP OP lookup table are consistent in MMCV, while the remaining two cases are not.

2. Functions modified by `force_fp32` are forced to run with `fp32`, which is not mandatory in PyTorch AMP. This operation does not affect accuracy, but may affect training speed in some algorithms.

## AMP Usage

### Existing Models AMP Usage

The way to turn on AMP training is very simple:

```python
# fp16 settings
fp16 = dict(loss_scale=512.) # indicates static scale
```

Just set the `fp16` parameter in the configuration to turn on AMP training, and like the official one, it supports static `scale` and dynamic `scale` modes.

```python
# indicates static scale
fp16 = dict(loss_scale=512.)
# indicates dynamic scale 
fp16 = dict(loss_scale='dynamic')
# Flexible dynamic scale via dictionary form
fp16 = dict(loss_scale=dict(init_scale=512.,mode='dynamic'))
```

### Custom Model AMP Usage

If the user adds a new model from scratch, the core is to pay attention to 2 steps.

1. you need to use the `force_fp32` decorator for the `Head` training calculation `Loss` and the test calculation `bbox` of the defined model
2. While applying `force_fp32`, the corresponding class must not be missing `self.fp16_enabled = False`

The `auto_fp16` decorator is written in the `base` class, so the user-defined model is not normally used.
