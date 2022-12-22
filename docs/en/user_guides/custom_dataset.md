# The whole process of custom dataset annotation+training+testing+deployment

In our daily work and study, we often encounter some tasks that need to train custom dataset. There are few scenarios in which open-source datasets can be used as online models, so we need to carry out a series of operations on our custom datasets to ensure that the models can be put into production and serve users.

```{SeeAlso}
The video of this document has been posted on Bilibili: [A nanny level tutorials for custom datasets from annotationt to deployment](https://www.bilibili.com/video/BV1RG4y137i5)
```

```{Note}
All instructions in this document are done on Linux and are fully available on Windows, only slightly different in commands and operations.
```

Default that you have completed the installation of MMYOLO, if not installed, please refer to the document \[start your first step\] (https://mmyolo.readthedocs.io/zh_CN/latest/get_started.html#id1) for installation.

In this tutorial, we will introduce the whole process from annotating custom dataset to final training, testing and deployment. The overview steps are as below:

1.  Prepare dataset: `tools/misc/download_dataset.py`
2.  Use the software of [labelme](https://github.com/wkentaro/labelme) to annotate: `demo/image_demo.py` + labelme
3.  Convert the dataset into COCO format: `tools/dataset_converters/labelme2coco.py`
4.  Split dataset:`tools/misc/coco_split.py`
5.  Creat a config file based on dataset
6.  Dataset visualization analysis: `tools/analysis_tools/dataset_analysis.py`
7.  Optimize Anchor size: `tools/analysis_tools/optimize_anchors.py`
8.  Visualization the data processing part of config: `tools/analysis_tools/browse_dataset.py`
9.  Train: `tools/train.py`
10. Inference: `demo/image_demo.py`
11. Deployment

```{Note}
After obtaining the model weight and the mAP of validation set, users need to deep analyse the  bad cases of incorrect predictions in order to optimize model. MMYOLO will add this function in the future. Expect.
```

Each step is described in detail below.

## 1. Prepare the customized dataset

- If you don't have your own dataset, or want to use a small dataset to run the whole process, you can use the 144 images `cat` dataset provided with this tutorial (the raw picture of this dataset is supplied by @RangeKing, cleaned by @PeterH0323). This `cat` dataset will be used as an example for the rest tutorial.

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205423220-c4b8f2fd-22ba-4937-8e47-1b3f6a8facd8.png" alt="cat dataset"/>
</div>

The download is also very simple, requiring only one command (dataset compression package size `217 MB`):

```shell
python tools/misc/download_dataset.py --dataset-name cat --save-dir ./data/cat --unzip --delete
```

This dataset is automatically downloaded to the `./data/cat` dir with the following directory structure:

```shell
.
└── ./data/cat
    ├── images # image files
    │    ├── image1.jpg
    │    ├── image2.png
    │    └── ...
    ├── labels # labelme files
    │    ├── image1.json
    │    ├── image2.json
    │    └── ...
    ├── annotations # annotated files of COCO
    │    ├── annotations_all.json # all labels of COCO
    │    ├── trainval.json # 80% labels of the dataset
    │    └── test.json # 20% labels of the dataset
    └── class_with_id.txt # id + class_name file
```

This dataset can be trained directly. You can remove everything **outside** the `images` dir if you want to go through the whole process.

- If you already have a dataset, you can compose it into the following structure:

```shell
.
└── $DATA_ROOT
    └── images
         ├── image1.jpg
         ├── image2.png
         └── ...
```

## 2. Use the software of labelme to annotate

In general, there are two annotation methods:

- Software or algorithmic assistance + manual correction (Recommend, reduce costs and speed up)
- Only manual annotation

```{Note}
At present, we also consider to access third-party libraries to support the integration of algorithm-assisted annotation and manual optimized annotation by calling MMYOLO inference API through GUI interface.
If you have any interest or ideas, please leave a comment in the issue or contact us directly!
```

### 2.1 Software or algorithmic assistance + manual correction

The principle is using the existing model to inference, and save the result as label file. Manually operating the software and loading the generated label files, you only need to check whether each image is correctly labeled and whether there are missing objects.【assistance + manual correction】you can save a lot of time in order to **reduce costs and speed up** by this way.

```{Note}
If the existing model doesn't have the categories defined in your dataset, such as COCO pre-trained model, you can manually annotate 100 images to train an initial model, and then software assistance.
```

The process is described below:

#### 2.1.1 Software or algorithmic assistance

MMYOLO provide model inference script `demo/image_demo.py`. Setting `--to-labelme` to generate labelme format label file:

```shell
python demo/image_demo.py img \
                          config \
                          checkpoint
                          [--out-dir OUT_DIR] \
                          [--device DEVICE] \
                          [--show] \
                          [--deploy] \
                          [--score-thr SCORE_THR] \
                          [--class-name CLASS_NAME]
                          [--to-labelme]
```

These include:

- `img`： image path, supported by dir, file, URL;
- `config`：config file path of model;
- `checkpoint`：weight file path of model;
- `--out-dir`：inference results saved in this dir, default as `./output`, if set this `--show` parameter, the detection results are not saved;
- `--device`：cumputing resources, including `CUDA`, `CPU` etc., default as `cuda:0`;
- `--show`：display the detection results, default as `False`；
- `--deploy`：whether to switch to deploy mode;
- `--score-thr`：confidence threshold, default as `0.3`;
- `--to-labelme`：whether to export label files in `labelme` format, shouldn't exist with the `--show` at the same time.

For example:

Here, we'll use YOLOv5-s as an example to help us label the 'cat' dataset we just downloaded. First, download the weights for YOLOv5-s:

```shell
mkdir work_dirs
wget https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth -P ./work_dirs
```

Since the COCO 80 dataset already includes the `cat` class, we can directly load the COCO pre-trained model for assistant annotation.

```shell
python demo/image_demo.py ./data/cat/images \
                          ./configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                          ./work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
                          --out-dir ./data/cat/labels \
                          --class-name cat \
                          --to-labelme
```

```{Tip}
- If your dataset needs to label with multiclass, you can use this `--class-name class1 class2` format;
- Removing the `--class-name` flag to output all classes.
```

the generated label files saved in `--out-dir`:

```shell
.
└── $OUT_DIR
    ├── image1.json
    ├── image1.json
    └── ...
```

Here is an example of the original image and it's
generating json file:

<div align=center>
  <img src="https://user-images.githubusercontent.com/25873202/205471430-dcc882dd-16bb-45e4-938f-6b62ab3dff19.jpg" alt="图片" width="45%"/>
  <img src="https://user-images.githubusercontent.com/25873202/205471559-643aecc8-7fa3-4fff-be51-2fb0a570fdd3.png" alt="图片" width="45%"/>
</div>

#### 2.1.2 Manual annotation

In this tutorial, we use [labelme](https://github.com/wkentaro/labelme) to annotate

- Install labelme

```shell
conda create -n labelme python=3.8
conda activate labelme
pip install labelme==5.1.1
```

- Start labelme

```shell
labelme ${image dir path (same as the previous step)} \
        --output ${the dir path of label file（same as --out-dir）} \
        --autosave \
        --nodata
```

These include:

- `--output`：saved path of labelme file. If there already exists label file of some images, it will be loaded;
- `--autosave`：auto-save label file, and some tedioys steps will be omitted.
- `--nodata`：doesn't store the base64 encoding of each image, so setting this flag will greatly reduce the size of the label file.

For example：

```shell
cd /path/to/mmyolo
labelme ./data/cat/images --output ./data/cat/labels --autosave --nodata
```

Type in command and labelme will start, and then check label. If labelme fails to start, type `export QT_DEBUG_PLUGINS=1` in command to see which libraries are missing and install it.

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205432185-54407d83-3cee-473f-8743-656da157cf80.png" alt="label UI"/>
</div>

```{warning}
Make sure to use `rectangle` with the shortcut `Ctrl + R` (see below).

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/204076212-86dab4fa-13dd-42cd-93d8-46b04b864449.png" alt="rectangle"/>
</div>
```

### 2.2 Only manual annotation

The procedure is the same as 【2.1.2 Manual annotation】, except that this is a direct labeling, there is no pre-generated label.

## 3. Convert the dataset into COCO format

### 3.1 Using scripts to convert

MMYOLO provides scripts to convert labelme labels to COCO labels

```shell
python tools/dataset_converters/labelme2coco.py --img-dir ${image dir path} \
                                                --labels-dir ${label dir location} \
                                                --out ${output COCO label json path} \
                                                [--class-id-txt ${class_with_id.txt path}]
```

These include:
`--class-id-txt`: is the `.txt` file of `id class_name` dataset:

- If not specified, the script will be generated automatically in the same directory as `--out`, and save it as `class_with_id.txt`;

- If specified, the script will read but not add or overwrite. It will also check if there are any other classes in the `.txt` file and will give you an error if there are any. Please check the `.txt` file and add the new class and its `id`.

An example `.txt` file looks like this (`id` start at `1`, just like COCO):

```text
1 cat
2 dog
3 bicycle
4 motorcycle

```

For example:

Coonsider the `cat` dataset for this tutorial:

```shell
python tools/dataset_converters/labelme2coco.py --img-dir ./data/cat/images \
                                                --labels-dir ./data/cat/labels \
                                                --out ./data/cat/annotations/annotations_all.json
```

For the `cat` dataset in this demo (note that we don't need to include the background class), we can see that the generated `class_with_id.txt` has only the `1` class:

```text
1 cat

```

### 3.2 Check the converted COCO label

Using the following command, we can display the COCO label on the image, which will verify that there are no problems with the conversion:

```shell
python tools/analysis_tools/browse_coco_json.py --img-dir ${image dir path} \
                                                --ann-file ${COCO label json path}
```

For example:

```shell
python tools/analysis_tools/browse_coco_json.py --img-dir ./data/cat/images \
                                                --ann-file ./data/cat/annotations/annotations_all.json
```

<div align=center>
<img alt="Image" src="https://user-images.githubusercontent.com/25873202/205429166-a6e48d20-c60b-4571-b00e-54439003ad3b.png">
</div>

```{SeeAlso}
See [Visualizing COCO label](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#coco) for more information on `tools/analysis_tools/browse_coco_json.py`.
```

## 4. Divide dataset into training set, validation set and test set

Usually, custom dataset is a large folder with full of images. We need to divide the dataset into training set, validation set and test set by ourselves. If the amount of data is small, we can not divide the validation set. Here's how the split script works:

```shell
python tools/misc/coco_split.py --json ${COCO label json path} \
                                --out-dir ${divide label json saved path} \
                                --ratios ${ratio of division} \
                                [--shuffle] \
                                [--seed ${random seed for division}]
```

These include:

- `--ratios`: ratio of division. If only 2 are set, the split is `trainval + test`, and if 3 are set, the split is `train + val + test`. Two formats are supported - integer and decimal:

  - Integer: divide the dataset in proportion after normalization. Example: `--ratio 2 1 1` (the code will convert to `0.5 0.25 0.25`) or `--ratio 3 1`（the code will convert to `0.75 0.25`）

  - Decimal: divide the dataset in proportion. **If the sum does not add up to 1, the script performs an automatic normalization correction.** Example: `--ratio 0.8 0.1 0.1` or `--ratio 0.8 0.2`

- `--shuffle`: whether to shuffle the dataset before splitting.

- `--seed`: the random seed of dataset division. If not set, this will be generated automatically.

For example:

```shell
python tools/misc/coco_split.py --json ./data/cat/annotations/annotations_all.json \
                                --out-dir ./data/cat/annotations \
                                --ratios 0.8 0.2 \
                                --shuffle \
                                --seed 10
```

<div align=center>
<img alt="Image" src="https://user-images.githubusercontent.com/25873202/205428346-5fdfbfca-0682-47aa-b0be-fa467cd0c5f8.png">
</div>

## 5. Create a new config file based on the dataset

Make sure the dataset directory looks like this:

```shell
.
└── $DATA_ROOT
    ├── annotations
    │    ├── trainval.json # only divide into trainval + test according to the above commands; If you use 3 groups to divide the ratio, here is train.json、val.json、test.json
    │    └── test.json
    ├── images
    │    ├── image1.jpg
    │    ├── image1.png
    │    └── ...
    └── ...
```

Since this is custom dataset, we need to create a new config and add some information we want to change.

About naming the new config:

- This config inherits from `yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py`;
- We will train the class `cat` from the dataset provided with this tutorial (if you are using your own dataset, you can define the class name of your own dataset);
- The GPU tested in this tutorial is 1 x 3080Ti with 12G video memory, and the computer memory is 32G. The maximum batch size for YOLOv5-s training is `batch size = 32` (see the Appendix for detailed machine information);
- Training epoch is `100 epoch`.

To sum up: you can name it `yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py` and place it into the dir of `configs/custom_dataset`.

Create a new directory named `custom_dataset` inside configs dir, and add config file with the following content:

<div align=center>
<img alt="Image" src="https://user-images.githubusercontent.com/25873202/205428358-e32fb455-480a-4f14-9613-e4cc3193fb4d.png">
</div>

```python
_base_ = '../yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

max_epochs = 100  # maximum epochs for training
data_root = './data/cat/'  # absolute path to the dataset dir
# data_root = '/root/workspace/mmyolo/data/cat/'  # absolute path to the dataset dir inside the Docker container

# the path of result save, can be omitted, omitted save file name is located under work_dirs with the same name of config file.
# If a config variable changes only part of its parameters, changing this variable will save the new training file elsewhere
work_dir = './work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat'

# load_from can specify a local path or URL, setting the URL will automatically download, because the above has been downloaded, we set the local path here
# Since this tutorial is fine-tuning on the cat dataset, we need to use `load_from` to load the pre-trained model from MMYOLO. This allows for faster convergence and accuracy
load_from = './work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

# According to your GPU situation, modify the batch size, and YOLOv5-s defaults to 8 cards x 16bs
train_batch_size_per_gpu = 32
train_num_workers = 4  # recommend to use train_num_workers = nGPU x 4

save_epoch_intervals = 2  # save weights every interval round

# According to your GPU situation, modify the base_lr, modification ratio is base_lr_default * (your_bs / default_bs)
base_lr = _base_.base_lr / 4

anchors = [  # the anchor has been updated according to the characteristics of dataset. The generation of anchor will be explained in the following section.
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]

class_name = ('cat', )  # according to the label information of class_with_id.txt, set the class_name
num_classes = len(class_name)
metainfo = dict(
    CLASSES=class_name,
    PALETTE=[(220, 20, 60)]  # the color of drawing, free to set
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=20,  # number of epochs to start validation.  Here 20 is set because the accuracy of the first 20 epochs is not high and the test is not meaningful, so it is skipped
    val_interval=save_epoch_intervals  # the test evaluation is performed  iteratively every val_interval round
)

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),

        # loss_cls is dynamically adjusted based on num_classes, but when num_classes = 1, loss_cls is always 0
        loss_cls=dict(loss_weight=0.5 *
                      (num_classes / 80 * 3 / _base_.num_det_layers))))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        # if the dataset is too small, you can use RepeatDataset, which repeats the current dataset n times per epoch, where 5 is set.
        times=5,
        dataset=dict(
            type=_base_.dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/trainval.json',
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=_base_.train_pipeline)))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/trainval.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/trainval.json')
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=base_lr))

default_hooks = dict(
    # set how many epochs to save the model, and the maximum number of models to save,`save_best` is also the best model (recommended).
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        max_keep_ckpts=5,
        save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    # logger output interval
    logger=dict(type='LoggerHook', interval=10))

```

```{Note}
We put an identical config file in `projects/misc/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py`. You can choose to copy to `configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py` to start training directly.
```

## 6. Visual analysis of datasets

The script `tools/analysis_tools/dataset_analysis.py` will helo you get a plot of your dataset. The script can generate four types of analysis graphs:

- A distribution plot showing categories and the number of bbox instances: `show_bbox_num`
- A distribution plot showing categories and the width and height of bbox instances: `show_bbox_wh`
- A distribution plot showing categories and the width/height ratio of bbox instances: `show_bbox_wh_ratio`
- A distribution plot showing categories and the area of bbox instances based on the area rule: `show_bbox_area`

Here's how the script works:

```shell
python tools/analysis_tools/dataset_analysis.py ${CONFIG} \
                                                [--val-dataset ${TYPE}] \
                                                [--class-name ${CLASS_NAME}] \
                                                [--area-rule ${AREA_RULE}] \
                                                [--func ${FUNC}] \
                                                [--out-dir ${OUT_DIR}]
```

For example:

Consider the config of `cat` dataset in this tutorial:

Check the distribution of the training data:

```shell
python tools/analysis_tools/dataset_analysis.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                                                --out-dir work_dirs/dataset_analysis_cat/train_dataset
```

Check the distribution of the validation data:

```shell
python tools/analysis_tools/dataset_analysis.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                                                --out-dir work_dirs/dataset_analysis_cat/val_dataset \
                                                --val-dataset
```

Effect (click on the image to view a larger image):

<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>A distribution plot showing categories and the area of bbox instances based on the area rule</b>
      </td>
      <td>
        <b>A distribution plot showing categories and the width and height of bbox instances</b>
      </td>
    </tr>
    <tr align="center" valign="center">
      <td>
        <img alt="YOLOv5CocoDataset_bbox_area" src="https://user-images.githubusercontent.com/25873202/206709093-1ed40f4e-cae3-4383-b120-79ad44c12312.jpg" width="60%">
      </td>
      <td>
        <img alt="YOLOv5CocoDataset_bbox_wh" src="https://user-images.githubusercontent.com/25873202/206709127-aebbb238-4af8-46c8-b71e-8540ed5f5de1.jpg" width="60%">
      </td>
    </tr>
    <tr align="center" valign="center">
      <td>
        <b>A distribution plot showing categories and the number of bbox instances</b>
      </td>
      <td>
        <b>A distribution plot showing categories and the width/height ratio of bbox instances</b>
      </td>
    </tr>
    <tr align="center" valign="center">
      <td>
        <img alt="YOLOv5CocoDataset_bbox_num" src="https://user-images.githubusercontent.com/25873202/206709108-8cee54f3-3102-4ca2-a10a-e4adb760881b.jpg" width="60%">
      </td>
      <td>
        <img alt="YOLOv5CocoDataset_bbox_ratio" src="https://user-images.githubusercontent.com/25873202/206709115-17aeba09-4ff1-4697-8842-94fbada6c428.jpg" width="60%">
      </td>
    </tr>
  </tbody>
</table>

```{Note}
Due to the cat dataset used in this tutorial is relatively small, we use RepeatDataset in config. The numbers shown are actually repeated five times. If you want a repeat-free analysis, you can change the `times` argument in RepeatDataset from `5` to `1` for now.
```

From the analysis output, we can conclude that the training set of the `cat` dataset used in this tutorial has the following characteristics:

- The images are all `large object`;
- The number of categories cat is `655`;
- The width and height ratio of bbox is mostly concentrated in `1.0 ~ 1.11`, the minimum ratio is `0.36` and the maximum ratio is `2.9`;
- The width of bbox is about `500 ~ 600` , and the height is about `500 ~ 600`.

```{SeeAlso}
See [Visualizing Dataset Analysis](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#id4) for more information on `tools/analysis_tools/dataset_analysis.py`
```

## 7. Optimize Anchor size

```{Warning}
This step only works for anchor-base models such as YOLOv5;

This step can be skipped for Anchor-free models, such as YOLOv6, YOLOX.
```

The `tools/analysis_tools/optimize_anchors.py` script supports three anchor generation methods from YOLO series: `k-means`, `Differential Evolution` and `v5-k-means`.

In this tutorial, we will use YOLOv5 for training, with an input size of `640 x 640`, and `v5-k-means` to optimize anchor:

```shell
python tools/analysis_tools/optimize_anchors.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                                                --algorithm v5-k-means \
                                                --input-shape 640 640 \
                                                --prior-match-thr 4.0 \
                                                --out-dir work_dirs/dataset_analysis_cat
```

```{Note}
Because this command uses the k-means clustering algorithm, there is some randomness, which is related to the initialization. Therefore, the Anchor obtained by each execution will be somewhat different, but it is generated based on the dataset passed in, so it will not have any adverse effects.
```

The calculated anchors are as follows:

<div align=center>
<img alt="Anchor" src="https://user-images.githubusercontent.com/25873202/205422434-1a68cded-b055-42e9-b01c-3e51f8f5ef81.png">
</div>

Modify the `anchors` variable in config file:

```python
anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]
```

```{SeeAlso}
See [Optimize Anchor Sizes](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#id8) for more information on `tools/analysis_tools/optimize_anchors.py`
```

## 8. Visualization the data processing part of config

The script `tools/analysis_tools/browse_dataset.py` allows you to visualize the data processing part of config directly in the window, with the option to save the visualization to a specific directory.

Let's use the config file we just created `configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py` to visualize the images. Each image lasts for `3` seconds, and the images are not saved:

```shell
python tools/analysis_tools/browse_dataset.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
                                              --show-interval 3
```

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205472078-c958e90d-8204-4c01-821a-8b6a006f05b2.png" alt="image" width="60%"/>
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205472197-8228c75e-6046-404a-89b4-ed55eeb2cb95.png" alt="image" width="60%"/>
</div>

```{SeeAlso}
See [Visualizing Datasets](https://mmyolo.readthedocs.io/zh_CN/latest/user_guides/useful_tools.html#id3) for more information on `tools/analysis_tools/browse_dataset.py`
```

## 9. Train

Here are three points to explain:

1. Training visualization
2. YOLOv5 model training
3. Switching YOLO model training

### 9.1 Training visualization

If you need to use a browser to visualize the training process, MMYOLO currently offers two ways [wandb](https://wandb.ai/site) and [TensorBoard](https://tensorflow.google.cn/tensorboard). Pick one according to your own situation (we'll expand support for more visualization backends in the future).

#### 9.1.1 wandb

Wandb visualization need registered in [website](https://wandb.ai/site), and in the https://wandb.ai/settings for wandb API Keys.

<div align=center>
<img src="https://cdn.vansin.top/img/20220913212628.png" alt="image"/>
</div>

Then install it from the command line:

```shell
pip install wandb
# After running wandb login, enter the API Keys obtained above, and the login is successful.
wandb login
```

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/206070473-201795e0-c81f-4247-842a-16d6acae0474.png" alt="推理图片"/>
</div>

Add the `wandb` configuration at the end of config file we just created, `configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py`.

```python
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])
```

#### 9.1.2 TensorBoard

Install Tensorboard environment

```shell
pip install tensorboard
```

Add the `tensorboard` configuration at the end of config file we just created, `configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py`.

```python
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
```

After running the training command, Tensorboard files will be generated in the visualization folder `work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat/${TIMESTAMP}/vis_data`. We can use Tensorboard to view the loss, learning rate, and coco/bbox_mAP visualizations from a web link by running the following command:

```shell
tensorboard --logdir=work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat
```

### 9.2 Perform training

Let's start the training with the following command (training takes about 2.5 hours) :

```shell
python tools/train.py configs/custom_dataset/yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py
```

If you have enabled wandb, you can log in to your account to view the details of this training in wandb:

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/206097557-7b10cf0f-8a16-4ba6-8563-b0a3cb149537.png" alt="Image"/>
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/206097706-7e131bf7-f3bf-43fb-9fe5-5589a324de69.png" alt="Image"/>
</div>

The following is `1 x 3080Ti`, `batch size = 32`, training `100 epoch` optimal precision weight `work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat/best_coco/bbox_mAP_epoch_98.pth` obtained accuracy (see Appendix for detailed machine information):

```shell
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.968
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.968
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.886
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.977
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.977
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.977

bbox_mAP_copypaste: 0.968 1.000 1.000 -1.000 -1.000 0.968
Epoch(val) [98][116/116]  coco/bbox_mAP: 0.9680  coco/bbox_mAP_50: 1.0000  coco/bbox_mAP_75: 1.0000  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.9680
```

```{Tip}
In general finetune best practice, it is recommended that backbone be left out of training and that the learning rate lr be scaled accordingly. However, in this tutorial, we found this approach can fall short to some extent. The possible reason is that the cat category is already in the COCO dataset, and the cat dataset used in this tutorial is relatively small
```

The following table shows the test accuracy of the MMYOLO YOLOv5 pre-trained model `yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth` without finetune on the cat dataset. It can be seen that the mAP of the `cat` category is only `0.866`, which improve to `0.968` after finetune, improved by '10.2%', which proves that the training was very successful:

```shell
+---------------+-------+--------------+-----+----------------+------+
| category      | AP    | category     | AP  | category       | AP   |
+---------------+-------+--------------+-----+----------------+------+
| person        | nan   | bicycle      | nan | car            | nan  |
| motorcycle    | nan   | airplane     | nan | bus            | nan  |
| train         | nan   | truck        | nan | boat           | nan  |
| traffic light | nan   | fire hydrant | nan | stop sign      | nan  |
| parking meter | nan   | bench        | nan | bird           | nan  |
| cat           | 0.866 | dog          | nan | horse          | nan  |
| sheep         | nan   | cow          | nan | elephant       | nan  |
| bear          | nan   | zebra        | nan | giraffe        | nan  |
| backpack      | nan   | umbrella     | nan | handbag        | nan  |
| tie           | nan   | suitcase     | nan | frisbee        | nan  |
| skis          | nan   | snowboard    | nan | sports ball    | nan  |
| kite          | nan   | baseball bat | nan | baseball glove | nan  |
| skateboard    | nan   | surfboard    | nan | tennis racket  | nan  |
| bottle        | nan   | wine glass   | nan | cup            | nan  |
| fork          | nan   | knife        | nan | spoon          | nan  |
| bowl          | nan   | banana       | nan | apple          | nan  |
| sandwich      | nan   | orange       | nan | broccoli       | nan  |
| carrot        | nan   | hot dog      | nan | pizza          | nan  |
| donut         | nan   | cake         | nan | chair          | nan  |
| couch         | nan   | potted plant | nan | bed            | nan  |
| dining table  | nan   | toilet       | nan | tv             | nan  |
| laptop        | nan   | mouse        | nan | remote         | nan  |
| keyboard      | nan   | cell phone   | nan | microwave      | nan  |
| oven          | nan   | toaster      | nan | sink           | nan  |
| refrigerator  | nan   | book         | nan | clock          | nan  |
| vase          | nan   | scissors     | nan | teddy bear     | nan  |
| hair drier    | nan   | toothbrush   | nan | None           | None |
+---------------+-------+--------------+-----+----------------+------+
```

```{SeeAlso}
For details on how to get the accuracy of the pre-trained weights, see the appendix【2. How to test the accuracy of dataset on pre-trained weights】
```
