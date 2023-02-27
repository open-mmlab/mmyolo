# 恢复训练

如果训练中途停止，在训练命令最后加上 `--resume` ,程序会自动从 `work_dirs` 中加载最新的权重文件恢复训练。命令如下：

```shell
python tools/train.py python ./tools/train.py ${CONFIG} --resume
```

具体例子如下：

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py --resume
```
