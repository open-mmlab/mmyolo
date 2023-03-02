# 恢复训练

恢复训练是指从之前某次训练保存下来的状态开始继续训练，这里的状态包括模型的权重、优化器和优化器参数调整策略的状态。

用户可以在训练命令最后加上 `--resume` 恢复训练，程序会自动从 `work_dirs` 中加载最新的权重文件恢复训练。如果 `work_dir` 中有最新的 checkpoint（例如该训练在上一次训练时被中断），则会从该 checkpoint 恢复训练，否则（例如上一次训练还没来得及保存 checkpoint 或者启动了新的训练任务）会重新开始训练。下面是一个恢复训练的示例:

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py --resume
```
