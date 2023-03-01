# 指定特定设备训练或推理

如果你有多张 GPU，比如 8 张，其编号分别为 `0, 1, 2, 3, 4, 5, 6, 7`，使用单卡训练或推理时会默认使用卡 0。如果想指定其他卡进行训练或推理，可以使用以下命令：

```shell
CUDA_VISIBLE_DEVICES=5 python ./tools/train.py ${CONFIG} #train
CUDA_VISIBLE_DEVICES=5 python ./tools/test.py ${CONFIG} ${CHECKPOINT_FILE} #test
```

如果设置`CUDA_VISIBLE_DEVICES`为 -1 或者一个大于 GPU 最大编号的数，比如 8，将会使用 CPU 进行训练或者推理。

如果你想使用其中几张卡并行训练，可以使用如下命令：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ${CONFIG} ${GPU_NUM}
```

这里 `GPU_NUM` 为 4。另外如果在一台机器上多个任务同时多卡训练，需要设置不同的端口，比如以下命令：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG} 4
```
