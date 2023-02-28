# Specify specific GPUs during training or inference

If you have multiple GPUs, such as 8 GPUs, numbered `0, 1, 2, 3, 4, 5, 6, 7`, GPU 0 will be used by default for training or inference. If you want to specify other GPUs for training or inference, you can use the following commands:

```shell
CUDA_VISIBLE_DEVICES=5 python ./tools/train.py ${CONFIG} #train
CUDA_VISIBLE_DEVICES=5 python ./tools/test.py ${CONFIG} ${CHECKPOINT_FILE} #test
```

If you set `CUDA_VISIBLE_DEVICES` to -1 or a number greater than the maximum GPU number, such as 8, the CPU will be used for training or inference.

If you want to use several of these GPUs to train in parallel, you can use the following command:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ${CONFIG} ${GPU_NUM}
```

Here the `GPU_NUM` is 4. In addition, if multiple tasks are trained in parallel on one machine and each task requires multiple GPUs, the PORT of each task need to be set differently to avoid communication conflict, like the following commands:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG} 4
```
