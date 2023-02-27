# 自动混合精度（AMP）训练

## AMP 运行流程
PyTorch 官方的 AMP 用法突出自动转换混合精度，而 MMCV 中则为了兼容两套 API，引入了相对较多的手动转化操作。混合精度训练的核心在于 `with autocast`，在该上下文内部，你无需特意考虑 `fp16` 的设置，内部会基于 OP 自动推断是使用 `fp16` 还是 `fp32`，而在 MMCV 的实现中，虽然也是调用 PyTorch 实现的 AMP，但是在 PyTorch 大于等于 1.6 版本上，也额外进行了强制类型转化(为什么要强制类型转换，因为目前 PyTorch AMP 在某些场景还是存在小问题，程序会报错)，在某些场合可能和官方的 AMP 行为有些区别，这个区别主要体现在两个方面：

1. 对于某些 OP (`addcdiv, addcmul, atan2, bilinear, cat, cross, dot, equal, index_put, stack, tensordot`)，例如 `cat` ，在 PyTorch AMP 中会以最宽范围的类型为准，例如 `dot` 操作，如果 `fp16` 和 `fp32`运算，那么输出类型是 `fp32`，如果都是 `fp16`，则会采用 `fp16` 运行，如果都是 `fp32`类型，则输出依然是 `fp32`，但是该 `fp32` 类型在 MMCV 中则会强制为 `fp16`运算。
简单来说，PyTorch AMP OP 查找表中的前两种情况在 MMCV 中表现是一致的，其余两种情况则没有保持一致，暂时没有发现对性能有影响，当然我们后续也会不断优化，尽可能和官方 AMP 保持一致。

2. 被 `force_fp32` 修饰的函数会强制采用 `fp32`运行，而在 PyTorch AMP 中没有这个强制要求。这个操作不会影响精度，但是在某些算法中可能会影响训练速度。

## AMP 使用方式

### 已有模型 AMP 使用方法

开启 AMP 训练的方式非常简单：

```python
# fp16 settings 
fp16 = dict(loss_scale=512.) # 表示静态 scale 
```

只要在配置中设置 fp16 参数就表示开启 AMP 训练，和官方一样，支持静态 scale 和动态 scale 模式。

```python
# 表示静态 scale 
fp16 = dict(loss_scale=512.) 
# 表示动态 scale 
fp16 = dict(loss_scale='dynamic')  
# 通过字典形式灵活开启动态 scale 
fp16 = dict(loss_scale=dict(init_scale=512.,mode='dynamic'))  
```

### 自定义模型 AMP 使用方法

如果用户重头新增了一个模型，核心就是要注意 2 个步骤：

1. 在所定义模型的 `Head` 训练算 `Loss` 和测试算 `bbox` 时候，需要使用 `force_fp32` 装饰器
2. 在应用 `force_fp32` 的同时，对应类一定不能少了 `self.fp16_enabled = False`

`auto_fp16` 装饰器由于在 `base` 类中会写，故用户自定义模型一般是不会使用到。