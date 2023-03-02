# 模型复杂度分析

我们提供了 `tools/analysis_tools/get_flops.py` 脚本来帮助进行 MMYOLO 系列中所有模型的复杂度分析。目前支持计算并输出给定模型的 parameters, activation 以及 flops；同时支持以网络结构或表格的形式打印输出每一层网络的复杂度信息。

调用命令如下：

```shell
python tools/analysis_tools/get_flops.py
    ${CONFIG_FILE} \                           # 配置文件路径
    [--shape ${IMAGE_SIZE}] \                  # 输入图像大小（int），默认取 640*640
    [--show-arch ${ARCH_DISPLAY}] \            # 以网络结构形式逐层展示复杂度信息
    [--not-show-table ${TABLE_DISPLAY}] \      # 以表格形式逐层展示复杂度信息
    [--cfg-options ${CFG_OPTIONS}]             # 配置文件参数修改选项
# [] 代表可选参数，实际输入命令行时，不用输入 []
```

接下来以 RTMDet 中的 `rtmdet_s_syncbn_fast_8xb32-300e_coco.py` 配置文件为例，详细展示该脚本的几种使用情形：

## 样例 1: 打印模型的 Flops 和 Parameters，并以表格形式展示每层网络复杂度

```shell
python tools/analysis_tools/get_flops.py  configs/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py
```

输出如下：

```python
==============================
Input shape: torch.Size([640, 640])
Model Flops: 14.835G
Model Parameters: 8.887M
==============================
```

| module                            | #parameters or shape | #flops  | #activations |
| :-------------------------------- | :------------------- | :------ | :----------: |
| model                             | 8.887M               | 14.835G |   35.676M    |
| backbone                          | 4.378M               | 5.416G  |   22.529M    |
| backbone.stem                     | 7.472K               | 0.765G  |    6.554M    |
| backbone.stem.0                   | 0.464K               | 47.514M |    1.638M    |
| backbone.stem.1                   | 2.336K               | 0.239G  |    1.638M    |
| backbone.stem.2                   | 4.672K               | 0.478G  |    3.277M    |
| backbone.stage1                   | 42.4K                | 0.981G  |    7.373M    |
| backbone.stage1.0                 | 18.56K               | 0.475G  |    1.638M    |
| backbone.stage1.1                 | 23.84K               | 0.505G  |    5.734M    |
| backbone.stage2                   | 0.21M                | 1.237G  |    4.915M    |
| backbone.stage2.0                 | 73.984K              | 0.473G  |    0.819M    |
| backbone.stage2.1                 | 0.136M               | 0.764G  |    4.096M    |
| backbone.stage3                   | 0.829M               | 1.221G  |    2.458M    |
| backbone.stage3.0                 | 0.295M               | 0.473G  |    0.41M     |
| backbone.stage3.1                 | 0.534M               | 0.749G  |    2.048M    |
| backbone.stage4                   | 3.29M                | 1.211G  |    1.229M    |
| backbone.stage4.0                 | 1.181M               | 0.472G  |    0.205M    |
| backbone.stage4.1                 | 0.657M               | 0.263G  |    0.307M    |
| backbone.stage4.2                 | 1.452M               | 0.476G  |    0.717M    |
| neck                              | 3.883M               | 4.366G  |    8.141M    |
| neck.reduce_layers.2              | 0.132M               | 52.634M |    0.102M    |
| neck.reduce_layers.2.conv         | 0.131M               | 52.429M |    0.102M    |
| neck.reduce_layers.2.bn           | 0.512K               | 0.205M  |      0       |
| neck.top_down_layers              | 0.491M               | 1.23G   |    4.506M    |
| neck.top_down_layers.0            | 0.398M               | 0.638G  |    1.638M    |
| neck.top_down_layers.1            | 92.608K              | 0.593G  |    2.867M    |
| neck.downsample_layers            | 0.738M               | 0.472G  |    0.307M    |
| neck.downsample_layers.0          | 0.148M               | 0.236G  |    0.205M    |
| neck.downsample_layers.1          | 0.59M                | 0.236G  |    0.102M    |
| neck.bottom_up_layers             | 1.49M                | 0.956G  |    2.15M     |
| neck.bottom_up_layers.0           | 0.3M                 | 0.48G   |    1.434M    |
| neck.bottom_up_layers.1           | 1.19M                | 0.476G  |    0.717M    |
| neck.out_layers                   | 1.033M               | 1.654G  |    1.075M    |
| neck.out_layers.0                 | 0.148M               | 0.945G  |    0.819M    |
| neck.out_layers.1                 | 0.295M               | 0.472G  |    0.205M    |
| neck.out_layers.2                 | 0.59M                | 0.236G  |    51.2K     |
| neck.upsample_layers              |                      | 1.229M  |      0       |
| neck.upsample_layers.0            |                      | 0.41M   |      0       |
| neck.upsample_layers.1            |                      | 0.819M  |      0       |
| bbox_head.head_module             | 0.625M               | 5.053G  |    5.006M    |
| bbox_head.head_module.cls_convs   | 0.296M               | 2.482G  |    2.15M     |
| bbox_head.head_module.cls_convs.0 | 0.295M               | 2.481G  |    2.15M     |
| bbox_head.head_module.cls_convs.1 | 0.512K               | 0.819M  |      0       |
| bbox_head.head_module.cls_convs.2 | 0.512K               | 0.205M  |      0       |
| bbox_head.head_module.reg_convs   | 0.296M               | 2.482G  |    2.15M     |
| bbox_head.head_module.reg_convs.0 | 0.295M               | 2.481G  |    2.15M     |
| bbox_head.head_module.reg_convs.1 | 0.512K               | 0.819M  |      0       |
| bbox_head.head_module.reg_convs.2 | 0.512K               | 0.205M  |      0       |
| bbox_head.head_module.rtm_cls     | 30.96K               | 86.016M |    0.672M    |
| bbox_head.head_module.rtm_cls.0   | 10.32K               | 65.536M |    0.512M    |
| bbox_head.head_module.rtm_cls.1   | 10.32K               | 16.384M |    0.128M    |
| bbox_head.head_module.rtm_cls.2   | 10.32K               | 4.096M  |     32K      |
| bbox_head.head_module.rtm_reg     | 1.548K               | 4.301M  |    33.6K     |
| bbox_head.head_module.rtm_reg.0   | 0.516K               | 3.277M  |    25.6K     |
| bbox_head.head_module.rtm_reg.1   | 0.516K               | 0.819M  |     6.4K     |
| bbox_head.head_module.rtm_reg.2   | 0.516K               | 0.205M  |     1.6K     |

## 样例 2：以网络结构形式逐层展示模型复杂度信息

```shell
python tools/analysis_tools/get_flops.py  configs/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py --show-arch
```

由于该网络结构复杂，输出较长。以下仅展示 bbox_head.head_module.rtm_reg 部分的输出：

```python
(rtm_reg): ModuleList(
        #params: 1.55K, #flops: 4.3M, #acts: 33.6K
        (0): Conv2d(
          128, 4, kernel_size=(1, 1), stride=(1, 1)
          #params: 0.52K, #flops: 3.28M, #acts: 25.6K
        )
        (1): Conv2d(
          128, 4, kernel_size=(1, 1), stride=(1, 1)
          #params: 0.52K, #flops: 0.82M, #acts: 6.4K
        )
        (2): Conv2d(
          128, 4, kernel_size=(1, 1), stride=(1, 1)
          #params: 0.52K, #flops: 0.2M, #acts: 1.6K
        )
```
