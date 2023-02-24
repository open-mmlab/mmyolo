# Output prediction results

If you want to save the prediction results as a specific file for offline evaluation, MMYOLO currently supports both json and pkl formats.

```{note}
The json file only save `image_id`, `bbox`, `score` and `category_id`. The json file can be read using the json library.
The pkl file holds more content than the json file, and also holds information such as the file name and size of the predicted image; the pkl file can be read using the pickle library. The pkl file can be read using the pickle library.
```

## Output into json file

If you want to output the prediction results as a json file, the command is as follows.

```shell
python tools/test.py {path_to_config} {path_to_checkpoint} --json-prefix {json_prefix}
```

The argument after `--json-prefix` should be a filename prefix (no need to enter the `.json` suffix) and can also contain a path. For a concrete example:

```shell
python tools/test.py configs\yolov5\yolov5_s-v61_syncbn_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --json-prefix work_dirs/demo/json_demo
```

Running the above command will output the `json_demo.bbox.json` file in the `work_dirs/demo` folder.

## Output into pkl file

If you want to output the prediction results as a pkl file, the command is as follows.

```shell
python tools/test.py {path_to_config} {path_to_checkpoint} --out {path_to_output_file}
```

The argument after `--out` should be a full filename (**must be** with a `.pkl` or `.pickle` suffix) and can also contain a path. For a concrete example:

```shell
python tools/test.py configs\yolov5\yolov5_s-v61_syncbn_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --out work_dirs/demo/pkl_demo.pkl
```

Running the above command will output the `pkl_demo.pkl` file in the `work_dirs/demo` folder.
