# 恢复训练

恢复训练是指从之前某次训练保存下来的状态开始继续训练，这里的状态包括模型的权重、优化器和优化器参数调整策略的状态。

## 自动恢复训练

用户可以设置 `Runner` 的 `resume` 参数开启自动恢复训练的功能。在启动训练时，设置 `Runner` 的 `resume` 等于 `True`，`Runner` 会从 `work_dir` 中加载最新的 checkpoint。如果 `work_dir` 中有最新的 checkpoint（例如该训练在上一次训练时被中断），则会从该 checkpoint 恢复训练，否则（例如上一次训练还没来得及保存 checkpoint 或者启动了新的训练任务）会重新开始训练。下面是一个开启自动恢复训练的示例

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
    resume=True,
)
runner.train()
```

## 指定 checkpoint 路径

如果希望指定恢复训练的路径，除了设置 `resume=True`，还需要设置 `load_from` 参数。需要注意的是，如果只设置了 `load_from` 而没有设置 `resume=True`，则只会加载 checkpoint 中的权重并重新开始训练，而不是接着之前的状态继续训练。

```python
runner = Runner(
    model=ResNet18(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader_cfg,
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=3),
    load_from='./work_dir/epoch_2.pth',
    resume=True,
)
runner.train()
```
