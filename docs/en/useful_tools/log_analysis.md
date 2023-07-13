# Log Analysis

## Curve plotting

`tools/analysis_tools/analyze_logs.py` in MMDetection plots loss/mAP curves given a training log file. Run `pip install seaborn` first to install the dependency.

```shell
mim run mmdet analyze_logs plot_curve \
    ${LOG} \                                     # path of train log in json format
    [--keys ${KEYS}] \                           # the metric that you want to plot, default to 'bbox_mAP'
    [--start-epoch ${START_EPOCH}]               # the epoch that you want to start, default to 1
    [--eval-interval ${EVALUATION_INTERVAL}] \   # the evaluation interval when training, default to 1
    [--title ${TITLE}] \                         # title of figure
    [--legend ${LEGEND}] \                       # legend of each plot, default to None
    [--backend ${BACKEND}] \                     # backend of plt, default to None
    [--style ${STYLE}] \                         # style of plt, default to 'dark'
    [--out ${OUT_FILE}]                          # the path of output file
# [] stands for optional parameters, when actually entering the command line, you do not need to enter []
```

Examples:

- Plot the classification loss of some run.

  ```shell
  mim run mmdet analyze_logs plot_curve \
      yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json \
      --keys loss_cls \
      --legend loss_cls
  ```

  <img src="https://user-images.githubusercontent.com/27466624/204747359-754555df-1f97-4d5c-87ca-9ad3a0badcce.png" width="600"/>

- Plot the classification and regression loss of some run, and save the figure to a pdf.

  ```shell
  mim run mmdet analyze_logs plot_curve \
      yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json \
      --keys loss_cls loss_bbox \
      --legend loss_cls loss_bbox \
      --out losses_yolov5_s.pdf
  ```

  <img src="https://user-images.githubusercontent.com/27466624/204748560-2d17ce4b-fb5f-4732-a962-329109e73aad.png" width="600"/>

- Compare the bbox mAP of two runs in the same figure.

  ```shell
  mim run mmdet analyze_logs plot_curve \
      yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json \
      yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739.log.json \
      --keys bbox_mAP \
      --legend yolov5_s yolov5_n \
      --eval-interval 10 # Note that the evaluation interval must be the same as during training. Otherwise, it will raise an error.
  ```

<img src="https://user-images.githubusercontent.com/27466624/204748704-21db9f9e-386e-449c-91c7-2ce3f8b51f24.png" width="600"/>

## Compute the average training speed

```shell
mim run mmdet analyze_logs cal_train_time \
    ${LOG} \                                # path of train log in json format
    [--include-outliers]                    # include the first value of every epoch when computing the average time
```

Examples:

```shell
mim run mmdet analyze_logs cal_train_time \
    yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json
```

The output is expected to be like the following.

```text
-----Analyze train time of yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json-----
slowest epoch 278, average time is 0.1705 s/iter
fastest epoch 300, average time is 0.1510 s/iter
time std over epochs is 0.0026
average iter time: 0.1556 s/iter
```
