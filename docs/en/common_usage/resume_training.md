# Resume training

Resume training means to continue training from the state saved from one of the previous trainings, where the state includes the model weights, the state of the optimizer and the optimizer parameter adjustment strategy.

The user can add `--resume` at the end of the training command to resume training, and the program will automatically load the latest weight file from `work_dirs` to resume training. If there is an updated checkpoint in `work_dir` (e.g. the training was interrupted during the last training), the training will be resumed from that checkpoint, otherwise (e.g. the last training did not have time to save the checkpoint or a new training task was started) the training will be restarted. Here is an example of resuming training:

```shell
python tools/train.py configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py --resume
```
