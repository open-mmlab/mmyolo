# 常见警告说明

本文档收集用户经常疑惑的警告信息说明，方便大家理解。

## xxx registry in mmyolo did not set import location

完整信息为 The xxx registry in mmyolo did not set import location. Fallback to call `mmyolo.utils.register_all_modules` instead.。
这个警告的含义说某个模块在导入时候发现没有设置导入的 location，导致无法确定其位置，因此会自动调用 `mmyolo.utils.register_all_modules` 触发包的导入。这个警告属于 MMEngine 中非常底层的模块警告，
用户理解起来可能比较困难，不过对大家使用没有任何影响，可以直接忽略。

## save_param_schedulers is true but self.param_schedulers is None

以 YOLOv5 算法为例，这是因为 YOLOv5 中重新写了参数调度器策略 `YOLOv5ParamSchedulerHook`，因此 MMEngine 中设计的 ParamScheduler 是没有使用的，但是 YOLOv5 配置中也没有设置 `save_param_schedulers` 为 False。
首先这个警告对性能和恢复训练没有任何影响，用户如果觉得这个警告会影响体验，可以设置 `default_hooks.checkpoint.save_param_scheduler` 为 False 或者训练时候通过命令行设置 `--cfg-options default_hooks.checkpoint.save_param_scheduler=False` 即可。

## The loss_cls will be 0. This is a normal phenomenon.

这个和具体算法有关。以 YOLOv5 为例，其分类 loss 是只考虑正样本的，如果类别是 1，那么分类 loss 和 obj loss 就是功能重复的了，因此在设计上当类别是 1 的时候 loss_cls 是不计算的，因此始终是 0，这是正常现象。

## The model and loaded state dict do not match exactly

这个警告是否会影响性能要根据进一步的打印信息来确定。如果是在微调模式下，由于用户自定义类别不一样无法加载 Head 模块的 COCO 预训练权重，这是一个正常现象，不会影响性能。
