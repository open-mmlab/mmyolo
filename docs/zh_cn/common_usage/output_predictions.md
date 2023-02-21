# 输出模型预测结果

如果想将预测结果保存为特定的文件，用于离线评估，目前 MMYOLO 支持 json 和 pkl 两种格式。

```{note}
json 文件仅保存 `image_id`、`bbox`、`score` 和 `category_id`； json 文件可以使用 json 库读取。
pkl 保存内容比 json 文件更多，还会保存预测图片的文件名和尺寸等一系列信息； pkl 文件可以使用 pickle 库读取。
```

## 输出为 json 文件

如果想将预测结果输出为 json 文件，则命令如下：

```shell
python tools/test.py ${CONFIG} ${CHECKPOINT} --json-prefix ${JSON_PREFIX}
```

`--json-prefix` 后的参数输入为文件名前缀（无需输入 `.json` 后缀），也可以包含路径。举一个具体例子：

```shell
python tools/test.py configs\yolov5\yolov5_s-v61_syncbn_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --json-prefix work_dirs/demo/json_demo
```

运行以上命令会在 `work_dirs/demo` 文件夹下，输出 `json_demo.bbox.json` 文件。

## 输出为 pkl 文件

如果想将预测结果输出为 pkl 文件，则命令如下：

```shell
python tools/test.py ${CONFIG} ${CHECKPOINT} --out ${OUTPUT_FILE} [--cfg-options ${OPTIONS [OPTIONS...]}]
```

`--out` 后的参数输入为完整文件名（**必须输入** `.pkl` 或 `.pickle` 后缀），也可以包含路径。举一个具体例子：

```shell
python tools/test.py configs\yolov5\yolov5_s-v61_syncbn_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --out work_dirs/demo/pkl_demo.pkl
```

运行以上命令会在 `work_dirs/demo` 文件夹下，输出 `pkl_demo.pkl` 文件。
