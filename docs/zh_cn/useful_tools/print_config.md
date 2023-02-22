# 打印完整配置文件

MMDetection 中的 `tools/misc/print_config.py` 脚本可将所有配置继承关系展开，打印相应的完整配置文件。调用命令如下：

```shell
mim run mmdet print_config \
    ${CONFIG} \                              # 需要打印的配置文件路径
    [--save-path] \                          # 保存文件路径，必须以 .py, .json 或者 .yml 结尾
    [--cfg-options ${OPTIONS [OPTIONS...]}]  # 通过命令行参数修改配置文件
```

样例：

```shell
mim run mmdet print_config \
    configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py \
    --save-path ./work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon_whole.py
```

运行以上命令，会将 `yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py` 继承关系展开后的配置文件保存到 `./work_dirs` 文件夹内的 `yolov5_s-v61_syncbn_fast_1xb4-300e_balloon_whole.py` 文件中。
