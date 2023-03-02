# 常见错误排除步骤

本文档收集用户经常碰到的常见错误情况，并提供详细的排查步骤。如果你发现阅读本文你没有找到正确的解决方案，请联系我们或者提 PR 进行更新。提 PR 请参考 [如何给 MMYOLO 贡献代码](../recommended_topics/contributing.md)

## xxx is not in the model registry

这个错误信息是指某个模块没有被注册到 model 中。 这个错误出现的原因非常多，典型的情况有：

1. 你新增的模块没有在类别前面加上注册器装饰器 @MODELS.register_module()
2. 虽然注册了，但是注册错了位置，例如你实际想注册到 MMYOLO 中，但是你导入的 MODELS 是 MMDet 包里面的
3. 你注册了且注册正确了，但是没有在对应的 `__init__.py` 中加入导致没有被导入
4. 以上 3 个步骤都确认没问题，但是你是新增 py 文件来自定义模块的却没有重新安装 MMYOLO 导致没有生效，此时你可以重新安装一遍，即使你是 -e 模式安装也需要重新安装
5. 如果你是在 mmyolo 包路径下新增了一个 package, 除上述步骤外，你还需要在 [register_all_modules](https://github.com/open-mmlab/mmyolo/blob/main/mmyolo/utils/setup_env.py#L8) 函数中增加其导包代码，否则该 package 不会被自动触发
6. 你的环境中有多个版本 MMYOLO，你注册的和实际运行的实际上不是同一套代码，导致没有生效。此时你可以在程序运行前输入 `PYTHONPATH="$(dirname $0)/..":$PYTHONPATH` 强行使用当前代码

## loss_bbox 始终为 0

该原因出现主要有两个原因：

1. 训练过程中没有 GT 标注数据
2. 参数设置不合理导致训练中没有正样本

第一种情况出现的概率更大。 `loss_bbox` 通常是只考虑正样本的 loss，如果训练中没有正样本则始终为 0。如果是第一种原因照成的 `loss_bbox` 始终为 0，那么通常意味着你配置不对，特别是 dataset 部分的配置不正确。
一个非常典型的情况是用户的 `dataset` 中 `metainfo` 设置不正确或者设置了但是没有传给 dataset 导致加载后没有找到对应类别的 GT Bbox 标注。 这种情况请仔细阅读我们提供的 [示例配置](https://github.com/open-mmlab/mmyolo/blob/main/projects/misc/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py#L27) 。
验证 dataset 配置是否正确的一个最直接的途径是运行 [browse_dataset 脚本](https://github.com/open-mmlab/mmyolo/blob/main/tools/analysis_tools/browse_dataset.py)，如果可视化效果正确则说明是正确的。

## MMCV 安装时间非常久

这通常意味着你在自己编译 MMCV 而不是直接下载使用我们提供的预编译包。 MMCV 中包括了大量的自定义的 CUDA 算子，如果从源码安装则需要非常久的时间去编译，并且由于其安装成功依赖于严格的底层环境信息，需要多个库的版本一致才可以。如果用户自己编译大概率会失败。
我们不推荐用户自己去编译 MMCV 而应该优先选择预编译包。如果你当前的环境中我们没有提供对应的预编译包，那么建议你可以快速换一个 Conda 环境，并安装有预编译包的 Torch。 以 torch1.8.0+cu102 为例，如果你想查看目前查看所有的预编译包，可以查看 https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html。

## 基于官方配置继承新建的配置出现 unexpected keyword argument

这通常是由于你没有删除 base 配置中的额外参数。 可以在你新建配置所修改的字典中增加 `_delete_=True` 删掉 base 中该类之前的所有参数。

## The testing results of the whole dataset is empty

这通常说明训练效果太差导致网络没有预测出任何符合阈值要求的检测框。 出现这种现象有多个原因，典型的为：

1. 当前为前几个 epoch，网络当前训练效果还较差，等后续训练久一点后可能就不会出现该警告了
2. 配置设置不正确，网络虽然正常训练但是实际上无效训练，例如前面的 `loss_bbox` 始终为 0 就会导致上述警告
3. 超参设置不合理

## ValueError: not enough values to unpack(expected 2, got 0)

这个错误通常是在 epoch 切换时候出现。这是 PyTorch 1.7 的已知问题，在 PyTorch 1.8+ 中已经修复。如果在 PyTorch 1.7 中想修复这个问题，可以简单的设置 dataloader 参数 `persistent_workers` 为 False。

## ValueError: need at least one array to concatenate

这个是一个非常常见的错误，可能出现在训练一开始或者训练正常但是评估时候。不管出现在何阶段，均说明你的配置不对：

1. 最常见的错误就是 `num_classes` 参数设置不对。在 MMYOLO 或者 MMDet 中大部分配置都是以 COCO 数据为例，因此配置中默认的 `num_classes` 是 80, 如果用户自定义数据集没有正确修改这个字段则会出现上述错误。
   MMYOLO 中有些算法配置会在多个模块中都需要 `num_classes` 参数，用户经常出现的错误就是仅仅修改了某一个地方的 `num_classes` 而没有将所有的 `num_classes` 都修改。想快速解决这个问题，可以使用 [print_config](https://github.com/open-mmlab/mmyolo/blob/main/tools/misc/print_config.py)
   脚本打印下全配置，然后全局搜索 `num_classes` 确认是否有没有修改的模块。
2. 该错误还可能会出现在对 `dataset.metainfo.classes` 参数设置不对造成的。当用户希望训练自己的数据集但是未能正确的修改 `dataset.metainfo.classes` 参数，而默认的使用 `COCO` 数据集中的类别时，且用户自定义数据集的所有类别不在 `COCO` 数据集的类别里就会出现该错误。这时需要用户核对并修改正确的 `dataset.metainfo.classes` 信息。

## 评估时候 IndexError: list index out of range

具体输出信息是

```text
  File "site-packages/mmdet/evaluation/metrics/coco_metric.py", line 216, in results2json
    data['category_id'] = self.cat_ids[label]
IndexError: list index out of range
```

可以看出是评估时候类别索引越界，这个通常的原因是配置中的 `num_classes` 设置不正确，默认的 `num_classes` 是 80，如果你自定义类别小于 80，那么就有可能出现类别越界。注意算法配置的 `num_classes` 一般会用到多个模块，你可能只改了某几个而漏掉了一些。想快速解决这个问题，可以使用 [print_config](https://github.com/open-mmlab/mmyolo/blob/main/tools/misc/print_config.py)
脚本打印下全配置，然后全局搜索 `num_classes` 确认是否有没有修改的模块。

## 训练中不打印 loss，但是程序依然正常训练和评估

这通常是因为一个训练 epoch 没有超过 50 个迭代，而 MMYOLO 中默认的打印间隔是 50。你可以修改 `default_hooks.logger.interval` 参数。

## GPU out of memory

1. 存在大量 ground truth boxes 或者大量 anchor 的场景，可能在 assigner 会 OOM。
2. 使用 --amp 来开启混合精度训练。
3. 你也可以尝试使用 MMDet 中的 AvoidCUDAOOM 来避免该问题。首先它将尝试调用 torch.cuda.empty_cache()。如果失败，将会尝试把输入类型转换到 FP16。如果仍然失败，将会把输入从 GPUs 转换到 CPUs 进行计算。这里提供了两个使用的例子：

```python
from mmdet.utils import AvoidCUDAOOM

output = AvoidCUDAOOM.retry_if_cuda_oom(some_function)(input1, input2)
```

你也可也使用 AvoidCUDAOOM 作为装饰器让代码遇到 OOM 的时候继续运行：

```python
from mmdet.utils import AvoidCUDAOOM

@AvoidCUDAOOM.retry_if_cuda_oom
def function(*args, **kwargs):
    ...
    return xxx
```

## Loss goes Nan

1. 检查数据的标注是否正常， 长或宽为 0 的框可能会导致回归 loss 变为 nan，一些小尺寸（宽度或高度小于 1）的框在数据增强后也会导致此问题。 因此，可以检查标注并过滤掉那些特别小甚至面积为 0 的框，并关闭一些可能会导致 0 面积框出现数据增强。
2. 降低学习率：由于某些原因，例如 batch size 大小的变化， 导致当前学习率可能太大。 您可以降低为可以稳定训练模型的值。
3. 延长 warm up 的时间：一些模型在训练初始时对学习率很敏感。
4. 添加 gradient clipping: 一些模型需要梯度裁剪来稳定训练过程。 你可以在 config 设置 `optim_wrapper.clip_grad=dict(max_norm=xx)`

## 训练中其他不符合预期或者错误

如果训练或者评估中出现了不属于上述描述的问题，由于原因不明，现提供常用的排除流程：

1. 首先确认配置是否正确，可以使用 [print_config](https://github.com/open-mmlab/mmyolo/blob/main/tools/misc/print_config.py) 脚本打印全部配置，如果运行成功则说明配置语法没有错误
2. 确认 COCO 格式的 json 标注是否正确，可以使用 [browse_coco_json.py](https://github.com/open-mmlab/mmyolo/blob/main/tools/misc/browse_coco_json.py) 脚本确认
3. 确认 dataset 部分配置是否正确，这一步骤几乎是必须要提前运行的，可以提前排查很多问题，可以使用 [browse_dataset.py](https://github.com/open-mmlab/mmyolo/blob/main/tools/misc/browse_dataset.py) 脚本确认
4. 如果以上 3 步都没有问题，那么出问题可能在 model 部分了。这个部分的排除没有特别的办法，你可以单独写一个脚本来仅运行 model 部分并通过调试来确认，如果对于 model 中多个模块的输入构建存在困惑，可以参考对应模块的单元测试写法
