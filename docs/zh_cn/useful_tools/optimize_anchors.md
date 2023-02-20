# 优化锚框尺寸

脚本 `tools/analysis_tools/optimize_anchors.py` 支持 YOLO 系列中三种锚框生成方式，分别是 `k-means`、`Differential Evolution`、`v5-k-means`.

## k-means

在 k-means 方法中，使用的是基于 IoU 表示距离的聚类方法，具体使用命令如下:

```shell
python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
                                                --algorithm k-means \
                                                --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
                                                --out-dir ${OUT_DIR}
```

## Differential Evolution

在 `Differential Evolution` 方法中，使用的是基于差分进化算法（简称 DE 算法）的聚类方式，其最小化目标函数为 `avg_iou_cost`，具体使用命令如下:

```shell
python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
                                                --algorithm DE \
                                                --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
                                                --out-dir ${OUT_DIR}
```

## v5-k-means

在 v5-k-means 方法中，使用的是 YOLOv5 中基于 `shape-match` 的聚类方式，具体使用命令如下:

```shell
python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
                                                --algorithm v5-k-means \
                                                --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
                                                --prior-match-thr ${PRIOR_MATCH_THR} \
                                                --out-dir ${OUT_DIR}
```
