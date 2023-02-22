# Print the whole config

`print_config.py` in MMDetection prints the whole config verbatim, expanding all its imports. The command is as following.

```shell
mim run mmdet print_config \
    ${CONFIG} \                              # path of the config file
    [--save-path] \                          # save path of whole config, suffixed with .py, .json or .yml
    [--cfg-options ${OPTIONS [OPTIONS...]}]  # override some settings in the used config
```

Examples:

```shell
mim run mmdet print_config \
    configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py \
    --save-path ./work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py
```

Running the above command will save the `yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py` config file with the inheritance relationship expanded to \`\`yolov5_s-v61_syncbn_fast_1xb4-300e_balloon_whole.py`in the`./work_dirs\` folder.
