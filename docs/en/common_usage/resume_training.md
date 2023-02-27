# Resume training

If training is stopped in the middle, add `--resume` at the end of the training command to automatically resume training by loading the latest weights from `work_dirs`. The command is as follows:

```shell
python tools/train.py python ./tools/train.py ${CONFIG} --resume
```

Specific examples are as follows:

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py --resume
```
