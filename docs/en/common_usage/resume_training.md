# Resume training

Resuming training means continuing training from the state saved from some previous training, where the state includes the model’s weights, the state of the optimizer and the state of parameter scheduler.

## Automatically resume training

Users can set the `resume` parameter of `Runner` to enable automatic training resumption. When `resume` is set to `True`, the Runner will try to resume from the latest checkpoint in `work_dir` automatically. If there is a latest checkpoint in `work_dir` (e.g. the training was interrupted during the last training), the training will be resumed from that checkpoint, otherwise (e.g. the last training did not have time to save the checkpoint or a new training task is started) the training will restart. Here is an example of how to enable automatic training resumption.

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

## Specify the checkpoint path

If you want to specify the path to resume training, you need to set `load_from` in addition to `resume=True`. Note that if only `load_from` is set without `resume=True`, then only the weights in the checkpoint will be loaded and training will be restarted, instead of continuing with the previous state.

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
