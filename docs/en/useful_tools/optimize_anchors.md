# Optimize anchors size

Script `tools/analysis_tools/optimize_anchors.py` supports three methods to optimize YOLO anchors including `k-means`
anchor cluster, `Differential Evolution` and `v5-k-means`.

## k-means

In k-means method, the distance criteria is based IoU, python shell as follow:

```shell
python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
                                                --algorithm k-means \
                                                --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
                                                --out-dir ${OUT_DIR}
```

## Differential Evolution

In differential_evolution method, based differential evolution algorithm, use `avg_iou_cost` as minimum target function, python shell as follow:

```shell
python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
                                                --algorithm DE \
                                                --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
                                                --out-dir ${OUT_DIR}
```

## v5-k-means

In v5-k-means method, clustering standard as same with YOLOv5 which use shape-match, python shell as follow:

```shell
python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
                                                --algorithm v5-k-means \
                                                --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
                                                --prior_match_thr ${PRIOR_MATCH_THR} \
                                                --out-dir ${OUT_DIR}
```
