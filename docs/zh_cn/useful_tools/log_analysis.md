# 日志分析

## 曲线图绘制

MMDetection 中的 `tools/analysis_tools/analyze_logs.py` 可利用指定的训练 log 文件绘制 loss/mAP 曲线图， 第一次运行前请先运行 `pip install seaborn` 安装必要依赖。

```shell
mim run mmdet analyze_logs plot_curve \
    ${LOG} \                                     # 日志文件路径
    [--keys ${KEYS}] \                           # 需要绘制的指标，默认为 'bbox_mAP'
    [--start-epoch ${START_EPOCH}]               # 起始的 epoch，默认为 1
    [--eval-interval ${EVALUATION_INTERVAL}] \   # 评估间隔，默认为 1
    [--title ${TITLE}] \                         # 图片标题，无默认值
    [--legend ${LEGEND}] \                       # 图例，默认为 None
    [--backend ${BACKEND}] \                     # 绘制后端，默认为 None
    [--style ${STYLE}] \                         # 绘制风格，默认为 'dark'
    [--out ${OUT_FILE}]                          # 输出文件路径
# [] 代表可选参数，实际输入命令行时，不用输入 []
```

样例：

- 绘制分类损失曲线图

  ```shell
  mim run mmdet analyze_logs plot_curve \
      yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json \
      --keys loss_cls \
      --legend loss_cls
  ```

  <img src="https://user-images.githubusercontent.com/27466624/204747359-754555df-1f97-4d5c-87ca-9ad3a0badcce.png" width="600"/>

- 绘制分类损失、回归损失曲线图，保存图片为对应的 pdf 文件

  ```shell
  mim run mmdet analyze_logs plot_curve \
      yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json \
      --keys loss_cls loss_bbox \
      --legend loss_cls loss_bbox \
      --out losses_yolov5_s.pdf
  ```

  <img src="https://user-images.githubusercontent.com/27466624/204748560-2d17ce4b-fb5f-4732-a962-329109e73aad.png" width="600"/>

- 在同一图像中比较两次运行结果的 bbox mAP

  ```shell
  mim run mmdet analyze_logs plot_curve \
      yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json \
      yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739.log.json \
      --keys bbox_mAP \
      --legend yolov5_s yolov5_n \
      --eval-interval 10 # 注意评估间隔必须和训练时设置的一致，否则会报错
  ```

<img src="https://user-images.githubusercontent.com/27466624/204748704-21db9f9e-386e-449c-91c7-2ce3f8b51f24.png" width="600"/>

## 计算平均训练速度

```shell
mim run mmdet analyze_logs cal_train_time \
    ${LOG} \                                # 日志文件路径
    [--include-outliers]                    # 计算时包含每个 epoch 的第一个数据
```

样例：

```shell
mim run mmdet analyze_logs cal_train_time \
    yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json
```

输出以如下形式展示：

```text
-----Analyze train time of yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json-----
slowest epoch 278, average time is 0.1705 s/iter
fastest epoch 300, average time is 0.1510 s/iter
time std over epochs is 0.0026
average iter time: 0.1556 s/iter
```
