# Automatic mixed precision（AMP）training

To enable Automatic Mixing Precision (AMP) training, add `--amp` to the end of the training command, which is as follows:

```shell
python tools/train.py python ./tools/train.py ${CONFIG} --amp
```

Specific examples are as follows:

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py --amp
```
