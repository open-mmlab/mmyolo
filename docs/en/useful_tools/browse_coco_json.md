# Visualize COCO labels

`tools/analysis_tools/browse_coco_json.py` is a script that can visualization to display the COCO label in the picture.

```shell
python tools/analysis_tools/browse_coco_json.py [--data-root ${DATA_ROOT}] \
                                                [--img-dir ${IMG_DIR}] \
                                                [--ann-file ${ANN_FILE}] \
                                                [--wait-time ${WAIT_TIME}] \
                                                [--disp-all] [--category-names CATEGORY_NAMES [CATEGORY_NAMES ...]] \
                                                [--shuffle]
```

If images and labels are in the same folder, you can specify `--data-root` to the folder, and then `--img-dir` and `--ann-file` to specify the relative path of the folder. The code will be automatically spliced.
If the image and label files are not in the same folder, you do not need to specify `--data-root`, but directly specify `--img-dir` and `--ann-file` of the absolute path.

E.g:

1. Visualize all categories of `COCO` and display all types of annotations such as `bbox` and `mask`:

```shell
python tools/analysis_tools/browse_coco_json.py --data-root './data/coco' \
                                                --img-dir 'train2017' \
                                                --ann-file 'annotations/instances_train2017.json' \
                                                --disp-all
```

If images and labels are not in the same folder, you can use a absolutely path:

```shell
python tools/analysis_tools/browse_coco_json.py --img-dir '/dataset/image/coco/train2017' \
                                                --ann-file '/label/instances_train2017.json' \
                                                --disp-all
```

2. Visualize all categories of `COCO`, and display only the `bbox` type labels, and shuffle the image to show:

```shell
python tools/analysis_tools/browse_coco_json.py --data-root './data/coco' \
                                                --img-dir 'train2017' \
                                                --ann-file 'annotations/instances_train2017.json' \
                                                --shuffle
```

3. Only visualize the `bicycle` and `person` categories of `COCO` and only the `bbox` type labels are displayed:

```shell
python tools/analysis_tools/browse_coco_json.py --data-root './data/coco' \
                                                --img-dir 'train2017' \
                                                --ann-file 'annotations/instances_train2017.json' \
                                                --category-names 'bicycle' 'person'
```

4. Visualize all categories of `COCO`, and display all types of label such as `bbox`, `mask`, and shuffle the image to show:

```shell
python tools/analysis_tools/browse_coco_json.py --data-root './data/coco' \
                                                --img-dir 'train2017' \
                                                --ann-file 'annotations/instances_train2017.json' \
                                                --disp-all \
                                                --shuffle
```
