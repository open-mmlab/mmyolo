# 自动混合精度（AMP）训练

如果要开启自动混合精度（AMP）训练，在训练命令最后加上 `--amp` 即可， 命令如下：

```shell
python tools/train.py python ./tools/train.py ${CONFIG} --amp
```

具体例子如下：

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py --amp
```
