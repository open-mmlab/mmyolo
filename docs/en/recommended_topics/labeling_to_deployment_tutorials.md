# Annotation-to-deployment workflow for custom dataset

In our daily work and study, we often encounter some tasks that need to train custom dataset. There are few scenarios in which open-source datasets can be used as online models, so we need to carry out a series of operations on our custom datasets to ensure that the models can be put into production and serve users.

```{SeeAlso}
The video of this document has been posted on Bilibili: [A nanny level tutorials for custom datasets from annotationt to deployment](https://www.bilibili.com/video/BV1RG4y137i5)
```

```{Note}
All instructions in this document are done on Linux and are fully available on Windows, only slightly different in commands and operations.
```

Default that you have completed the installation of MMYOLO, if not installed, please refer to the document [GET STARTED](https://mmyolo.readthedocs.io/en/latest/get_started.html) for installation.

In this tutorial, we will introduce the whole process from annotating custom dataset to final training, testing and deployment. The overview steps are as below:

01. Prepare dataset: `tools/misc/download_dataset.py`
02. Use the software of [labelme](https://github.com/wkentaro/labelme) to annotate: `demo/image_demo.py` + labelme
03. Convert the dataset into COCO format: `tools/dataset_converters/labelme2coco.py`
04. Split dataset:`tools/misc/coco_split.py`
05. Creat a config file based on dataset
06. Dataset visualization analysis: `tools/analysis_tools/dataset_analysis.py`
07. Optimize Anchor size: `tools/analysis_tools/optimize_anchors.py`
08. Visualization the data processing part of config: `tools/analysis_tools/browse_dataset.py`
09. Train: `tools/train.py`
10. Inference: `demo/image_demo.py`
11. Deployment

```{Note}
After obtaining the model weight and the mAP of validation set, users need to deep analyse the  bad cases of incorrect predictions in order to optimize model. MMYOLO will add this function in the future. Expect.
```

Each step is described in detail below.

## 1. Prepare custom dataset

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

Here is an example of the original image and it's generating json file:

<div align=center>
  <img src="https://user-images.githubusercontent.com/25873202/205471430-dcc882dd-16bb-45e4-938f-6b62ab3dff19.jpg" alt="Image" width="45%"/>
  <img src="https://user-images.githubusercontent.com/25873202/205471559-643aecc8-7fa3-4fff-be51-2fb0a570fdd3.png" alt="Image" width="45%"/>
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
See [Visualizing COCO label](https://mmyolo.readthedocs.io/en/latest/user_guides/useful_tools.html#coco) for more information on `tools/analysis_tools/browse_coco_json.py`.
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
data_root = './data/cat/'  # absolute path to the dataset directory
# data_root = '/root/workspace/mmyolo/data/cat/'  # absolute path to the dataset dir inside the Docker container

# the path of result save, can be omitted, omitted save file name is located under work_dirs with the same name of config file.
# If a config variable changes only part of its parameters, changing this variable will save the new training file elsewhere
work_dir = './work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_cat'

# load_from can specify a local path or URL, setting the URL will automatically download, because the above has been downloaded, we set the local path here
# since this tutorial is fine-tuning on the cat dataset, we need to use `load_from` to load the pre-trained model from MMYOLO. This allows for faster convergence and accuracy
load_from = './work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

# according to your GPU situation, modify the batch size, and YOLOv5-s defaults to 8 cards x 16bs
train_batch_size_per_gpu = 32
train_num_workers = 4  # recommend to use train_num_workers = nGPU x 4

save_epoch_intervals = 2  # save weights every interval round

# according to your GPU situation, modify the base_lr, modification ratio is base_lr_default * (your_bs / default_bs)
base_lr = _base_.base_lr / 4

anchors = [  # the anchor has been updated according to the characteristics of dataset. The generation of anchor will be explained in the following section.
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]

class_name = ('cat', )  # according to the label information of class_with_id.txt, set the class_name
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60)]  # the color of drawing, free to set
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
See [Visualizing Dataset Analysis](https://mmyolo.readthedocs.io/en/latest/user_guides/useful_tools.html#id4) for more information on `tools/analysis_tools/dataset_analysis.py`
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
See [Optimize Anchor Sizes](https://mmyolo.readthedocs.io/en/latest/user_guides/useful_tools.html#id8) for more information on `tools/analysis_tools/optimize_anchors.py`
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
See [Visualizing Datasets](https://mmyolo.readthedocs.io/en/latest/user_guides/useful_tools.html#id3) for more information on `tools/analysis_tools/browse_dataset.py`
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
<img src="https://user-images.githubusercontent.com/25873202/206070473-201795e0-c81f-4247-842a-16d6acae0474.png" alt="Image"/>
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

### 9.3 Switch other models in MMYOLO

MMYOLO integrates multiple YOLO algorithms, which makes switching between YOLO models very easy. There is no need to reacquaint with a new repo. You can easily switch between YOLO models by simply modifying the config file:

1. Create a new config file
2. Download the pre-trained weights
3. Starting training

Let's take YOLOv6-s as an example.

1. Create a new config file：

```python
_base_ = '../yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py'

max_epochs = 100  # maximum of training epoch
data_root = './data/cat/'  # absolute path to the dataset directory

# the path of result save, can be omitted, omitted save file name is located under work_dirs with the same name of config file.
# If a config variable changes only part of its parameters, changing this variable will save the new training file elsewhere
work_dir = './work_dirs/yolov6_s_syncbn_fast_1xb32-100e_cat'

# load_from can specify a local path or URL, setting the URL will automatically download, because the above has been downloaded, we set the local path here
# since this tutorial is fine-tuning on the cat dataset, we need to use `load_from` to load the pre-trained model from MMYOLO. This allows for faster convergence and accuracy
load_from = './work_dirs/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth'  # noqa

# according to your GPU situation, modify the batch size, and YOLOv6-s defaults to 8 cards x 32bs
train_batch_size_per_gpu = 32
train_num_workers = 4  # recommend to use  train_num_workers = nGPU x 4

save_epoch_intervals = 2  # save weights every interval round

# according to your GPU situation, modify the base_lr, modification ratio is base_lr_default * (your_bs / default_bs)
base_lr = _base_.base_lr / 8

class_name = ('cat', )  # according to the label information of class_with_id.txt, set the class_name
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60)]  # the color of drawing, free to set
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=20,  # number of epochs to start validation.  Here 20 is set because the accuracy of the first 20 epochs is not high and the test is not meaningful, so it is skipped
    val_interval=save_epoch_intervals,  # the test evaluation is performed  iteratively every val_interval round
    dynamic_intervals=[(max_epochs - _base_.num_last_epochs, 1)]
)

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        initial_assigner=dict(num_classes=num_classes),
        assigner=dict(num_classes=num_classes))
)

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

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - _base_.num_last_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
]

```

```{Note}
Similarly, We put an identical config file in `projects/misc/custom_dataset/yolov6_s_syncbn_fast_1xb32-100e_cat.py`. You can choose to copy to `configs/custom_dataset/yolov6_s_syncbn_fast_1xb32-100e_cat.py` to start training directly.

Even though the new config looks like a lot of stuff, it's actually a lot of duplication. You can use a comparison software to see that most of the configuration is identical to 'yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py'. Because the two config files need to inherit from different config files, you still need to add the necessary configuration.
```

2. Download the pre-trained weights

```bash
wget https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth -P work_dirs/
```

3. Starting training

```shell
python tools/train.py configs/custom_dataset/yolov6_s_syncbn_fast_1xb32-100e_cat.py
```

In my experiments, the best model is `work_dirs/yolov6_s_syncbn_fast_1xb32-100e_cat/best_coco/bbox_mAP_epoch_96.pth`,which accuracy is as follows:

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.987
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.987
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.895
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.989
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.989
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.989

bbox_mAP_copypaste: 0.987 1.000 1.000 -1.000 -1.000 0.987
Epoch(val) [96][116/116]  coco/bbox_mAP: 0.9870  coco/bbox_mAP_50: 1.0000  coco/bbox_mAP_75: 1.0000  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.9870
```

The above demonstrates how to switch models in MMYOLO, you can quickly compare the accuracy of different models, and the model with high accuracy can be put into production. In my experiment, the best accuracy of YOLOv6 `0.9870` is `1.9 %` higher than the best accuracy of YOLOv5 `0.9680` , so we will use YOLOv6 for explanation.

## 10. Inference

Using the best model for inference, the best model path in the following command is `./work_dirs/yolov6_s_syncbn_fast_1xb32-100e_cat/best_coco/bbox_mAP_epoch_96.pth`, please modify the best model path you trained.

```shell
python demo/image_demo.py ./data/cat/images \
                          ./configs/custom_dataset/yolov6_s_syncbn_fast_1xb32-100e_cat.py \
                          ./work_dirs/yolov6_s_syncbn_fast_1xb32-100e_cat/best_coco/bbox_mAP_epoch_96.pth \
                          --out-dir ./data/cat/pred_images
```

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/204773727-5d3cbbad-1265-45a0-822a-887713555049.jpg" alt="Image"/>
</div>

```{Tip}
If the inference result is not ideal, here are two cases:

1. Model underfitting:

   First, we need to determine if there is not enough training epochs resulting in underfitting. If there is not enough training, we need to change the `max_epochs` and `work_dir` parameters in the config file, or create a new config file named as above and start the training again.

2. The dataset needs to be optimized:
   If adding epochs still doesn't work, we can increase the number of datasets and re-examine and refine the annotations of the dataset before retraining.
```

## 11. Deployment

MMYOLO provides two deployment options:

1. [MMDeploy](https://github.com/open-mmlab/mmdeploy) framework for deployment
2. Using `projects/easydeploy` to deployment

### 11.1 MMDeploy framework for deployment

Considering that the wide variety of machine deployments, there are many times when a local machine will work, but not in production. Here, we recommended to use Docker, so that the environment can be deployed once and used for life, saving the time of operation and maintenance to build the environment and deploy production.

In this part, we will introduce the following steps:

1. Building a Docker image
2. Creating a Docker container
3. Transforming TensorRT models
4. Deploying model and performing inference

```{SeeAlso}
If you are not familiar with Docker, you can refer to the MMDeploy [source manual installation].(https://mmdeploy.readthedocs.io/en/latest/01-how-to-build/build_from_source.html) file to compile directly locally. Once installed, you can skip to【11.1.3 Transforming TensorRT models】
```

#### 11.1.1 Building a Docker image

```shell
git clone -b dev-1.x https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
docker build docker/GPU/ -t mmdeploy:gpu --build-arg USE_SRC_INSIDE=true
```

Where `USE_SRC_INSIDE=true` is to pull the basis after switching the domestic source, the build speed will be faster.

After executing the script, the build will start, which will take a while:

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205482447-329186c8-eba3-443f-b1fa-b33c2ab3d5da.png" alt="Image"/>
</div>

#### 11.1.2 Creating a Docker container

```shell
export MMYOLO_PATH=/path/to/local/mmyolo # write the path to MMYOLO on your machine to an environment variable
docker run --gpus all --name mmyolo-deploy -v ${MMYOLO_PATH}:/root/workspace/mmyolo -it mmdeploy:gpu /bin/bash
```

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205536974-1eeb2901-9b14-4851-9c96-5046cd05f171.png" alt="Image"/>
</div>

You can see your local MMYOLO environment mounted inside the container

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205537473-0afc16c3-c6d4-451a-96d7-1a2388341b60.png" alt="Image"/>
</div>

```{SeeAlso}
You can read more about this in the MMDeploy official documentation [Using Docker Images](https://mmdeploy.readthedocs.io/en/latest/01-how-to-build/build_from_docker.html#docker)
```

#### 11.1.3 Transforming TensorRT models

The first step is to install MMYOLO and `pycuda` in a Docker container：

```shell
export MMYOLO_PATH=/root/workspace/mmyolo # path in the image, which doesn't need to modify
cd ${MMYOLO_PATH}
export MMYOLO_VERSION=$(python -c "import mmyolo.version as v; print(v.__version__)")  # Check the version number of MMYOLO used for training
echo "Using MMYOLO ${MMYOLO_VERSION}"
mim install --no-cache-dir mmyolo==${MMYOLO_VERSION}
pip install --no-cache-dir pycuda==2022.2
```

Performing model transformations

```shell
cd /root/workspace/mmdeploy
python ./tools/deploy.py \
    ${MMYOLO_PATH}/configs/deploy/detection_tensorrt-fp16_dynamic-192x192-960x960.py \
    ${MMYOLO_PATH}/configs/custom_dataset/yolov6_s_syncbn_fast_1xb32-100e_cat.py \
    ${MMYOLO_PATH}/work_dirs/yolov6_s_syncbn_fast_1xb32-100e_cat/best_coco/bbox_mAP_epoch_96.pth \
    ${MMYOLO_PATH}/data/cat/images/mmexport1633684751291.jpg \
    --test-img ${MMYOLO_PATH}/data/cat/images/mmexport1633684751291.jpg \
    --work-dir ./work_dir/yolov6_s_syncbn_fast_1xb32-100e_cat_deploy_dynamic_fp16 \
    --device cuda:0 \
    --log-level INFO \
    --show \
    --dump-info
```

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/206736259-72b76698-cba4-4472-909d-0fd866b45d55.png" alt="Image"/>
</div>

Wait for a few minutes, `All process success.` appearance indicates success:

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/206736030-3b702929-4fcb-4cec-a6ce-f22a94777f6c.png" alt="Image"/>
</div>

Looking at the exported path, you can see the file structure as shown in the following screenshot:

```shell
$WORK_DIR
  ├── deploy.json
  ├── detail.json
  ├── end2end.engine
  ├── end2end.onnx
  └── pipeline.json
```

```{SeeAlso}
For a detailed description of transforming models, see [How to Transform Models](https://mmdeploy.readthedocs.io/en/latest/02-how-to-run/convert_model.html)
```

#### 11.1.4 Deploying model and performing inference

We need to change the `data_root` in `${MMYOLO_PATH}/configs/custom_dataset/yolov6_s_syncbn_fast_1xb32-100e_cat.py` to the path in the Docker container:

```python
data_root = '/root/workspace/mmyolo/data/cat/'  # absolute path of the dataset dir in the Docker container.
```

Execute speed and accuracy tests:

```shell
python tools/test.py \
    ${MMYOLO_PATH}/configs/deploy/detection_tensorrt-fp16_dynamic-192x192-960x960.py \
    ${MMYOLO_PATH}/configs/custom_dataset/yolov6_s_syncbn_fast_1xb32-100e_cat.py \
    --model ./work_dir/yolov6_s_syncbn_fast_1xb32-100e_cat_deploy_dynamic_fp16/end2end.engine \
    --speed-test \
    --device cuda
```

The speed test is as follows, we can see that the average inference speed is `24.10ms`, which is a speed improvement compared to PyTorch inference, but also reduce lots of video memory usage:

```shell
Epoch(test) [ 10/116]    eta: 0:00:20  time: 0.1919  data_time: 0.1330  memory: 12
Epoch(test) [ 20/116]    eta: 0:00:15  time: 0.1220  data_time: 0.0939  memory: 12
Epoch(test) [ 30/116]    eta: 0:00:12  time: 0.1168  data_time: 0.0850  memory: 12
Epoch(test) [ 40/116]    eta: 0:00:10  time: 0.1241  data_time: 0.0940  memory: 12
Epoch(test) [ 50/116]    eta: 0:00:08  time: 0.0974  data_time: 0.0696  memory: 12
Epoch(test) [ 60/116]    eta: 0:00:06  time: 0.0865  data_time: 0.0547  memory: 16
Epoch(test) [ 70/116]    eta: 0:00:05  time: 0.1521  data_time: 0.1226  memory: 16
Epoch(test) [ 80/116]    eta: 0:00:04  time: 0.1364  data_time: 0.1056  memory: 12
Epoch(test) [ 90/116]    eta: 0:00:03  time: 0.0923  data_time: 0.0627  memory: 12
Epoch(test) [100/116]    eta: 0:00:01  time: 0.0844  data_time: 0.0583  memory: 12
[tensorrt]-110 times per count: 24.10 ms, 41.50 FPS
Epoch(test) [110/116]    eta: 0:00:00  time: 0.1085  data_time: 0.0832  memory: 12
```

Accuracy test is as follows. This configuration uses FP16 format inference, which has some drop points, but it is faster and uses less video memory:

```shell
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.954
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.975
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.954
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.860
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.965
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.965
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.965

INFO - bbox_mAP_copypaste: 0.954 1.000 0.975 -1.000 -1.000 0.954
INFO - Epoch(test) [116/116]  coco/bbox_mAP: 0.9540  coco/bbox_mAP_50: 1.0000  coco/bbox_mAP_75: 0.9750  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.9540
```

Deployment model and inference demonstration:

```{Note}
You can use the MMDeploy SDK for deployment and use C++ to further improve inference speed.
```

```shell
cd ${MMYOLO_PATH}/demo
python deploy_demo.py \
    ${MMYOLO_PATH}/data/cat/images/mmexport1633684900217.jpg \
    ${MMYOLO_PATH}/configs/custom_dataset/yolov6_s_syncbn_fast_1xb32-100e_cat.py \
    /root/workspace/mmdeploy/work_dir/yolov6_s_syncbn_fast_1xb32-100e_cat_deploy_dynamic_fp16/end2end.engine \
    --deploy-cfg ${MMYOLO_PATH}/configs/deploy/detection_tensorrt-fp16_dynamic-192x192-960x960.py \
    --out-dir ${MMYOLO_PATH}/work_dirs/deploy_predict_out \
    --device cuda:0 \
    --score-thr 0.5
```

```{Warning}
The script `deploy_demo.py` doesn't achieve batch inference, and the pre-processing code needs to be improved. It cannot fully show the inference speed at the moment, only demonstrate the inference results. we will optimize in the future. Expect!
```

After executing, you can see the inference image results in `--out-dir` :

<div align=center>
<img src="https://user-images.githubusercontent.com/25873202/205815829-6f85e655-722a-47c8-9e23-2a74437c0923.jpg" alt="Image"/>
</div>

```{Note}
You can also use other optimizations like increasing batch size, int8 quantization, etc.
```

#### 11.1.5 Save and load the Docker container

It would be a waste of time to build a docker image every time. At this point you can consider using docker's packaging api for packaging and loading.

```shell
# save, the result tar package can be placed on mobile hard disk
docker save mmyolo-deploy > mmyolo-deploy.tar

# load image to system
docker load < /path/to/mmyolo-deploy.tar
```

### 11.2 Using `projects/easydeploy` to deploy

```{SeeAlso}
See [deployment documentation](https://github.com/open-mmlab/mmyolo/blob/dev/projects/easydeploy/README.md) for details.
```

TODO: This part will be improved in the next version...

## Appendix

### 1. The detailed environment for training the machine in this tutorial is as follows:

```shell
sys.platform: linux
Python: 3.9.13 | packaged by conda-forge | (main, May 27 2022, 16:58:50) [GCC 10.3.0]
CUDA available: True
numpy_random_seed: 2147483648
GPU 0: NVIDIA GeForce RTX 3080 Ti
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 11.5, V11.5.119
GCC: gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
PyTorch: 1.10.0
PyTorch compiling details: PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) oneAPI Math Kernel Library Version 2021.4-Product Build 20210904 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.2.3 (Git Hash 7336ca9f055cf1bfa13efb658fe15dc9b41f0740)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.3
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;
                             arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;
                             -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;
                             arch=compute_86,code=sm_86;-gencode;arch=compute_37,code=compute_37
  - CuDNN 8.2
  - Magma 2.5.2
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.3, CUDNN_VERSION=8.2.0,
                    CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden
                    -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK
                    -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra
                    -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas
                    -Wno-sign-compare -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic
                    -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new
                    -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format
                    -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1,
                    TORCH_VERSION=1.10.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON,
                    USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON,

TorchVision: 0.11.0
OpenCV: 4.6.0
MMEngine: 0.3.1
MMCV: 2.0.0rc3
MMDetection: 3.0.0rc3
MMYOLO: 0.2.0+cf279a5
```

### 2. How to test the accuracy of our dataset on the pre-trained weights:

```{Warning}
Premise: The class is in the COCO 80 class!
```

In this part, we will use the `cat` dataset as an example, using:

- config file: `configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py`
- weight `yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth`

1. modify the path in config file

Because `configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py` is inherited from `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py`. Therefore, you can mainly modify the `configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py` file.

| before modification                                                            | after modification                                                          |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `data_root = 'data/coco/'`                                                     | `data_root = './data/cat/'`                                                 |
| `ann_file='annotations/instances_train2017.json'`                              | `ann_file='annotations/trainval.json'`                                      |
| data_prefix=dict(img='train2017/')\`                                           | `data_prefix=dict(img='images/')`                                           |
| `val_evaluator` of `ann_file=data_root + 'annotations/instances_val2017.json'` | `val_evaluator` of `dict(ann_file=data_root + 'annotations/trainval.json')` |

2. modify label

```{note}
It is recommended to make a copy of the label directly to prevent damage to original label
```

Change the `categories` in `trainval.json` to COCO's original:

```json
  "categories": [{"supercategory": "person","id": 1,"name": "person"},{"supercategory": "vehicle","id": 2,"name": "bicycle"},{"supercategory": "vehicle","id": 3,"name": "car"},{"supercategory": "vehicle","id": 4,"name": "motorcycle"},{"supercategory": "vehicle","id": 5,"name": "airplane"},{"supercategory": "vehicle","id": 6,"name": "bus"},{"supercategory": "vehicle","id": 7,"name": "train"},{"supercategory": "vehicle","id": 8,"name": "truck"},{"supercategory": "vehicle","id": 9,"name": "boat"},{"supercategory": "outdoor","id": 10,"name": "traffic light"},{"supercategory": "outdoor","id": 11,"name": "fire hydrant"},{"supercategory": "outdoor","id": 13,"name": "stop sign"},{"supercategory": "outdoor","id": 14,"name": "parking meter"},{"supercategory": "outdoor","id": 15,"name": "bench"},{"supercategory": "animal","id": 16,"name": "bird"},{"supercategory": "animal","id": 17,"name": "cat"},{"supercategory": "animal","id": 18,"name": "dog"},{"supercategory": "animal","id": 19,"name": "horse"},{"supercategory": "animal","id": 20,"name": "sheep"},{"supercategory": "animal","id": 21,"name": "cow"},{"supercategory": "animal","id": 22,"name": "elephant"},{"supercategory": "animal","id": 23,"name": "bear"},{"supercategory": "animal","id": 24,"name": "zebra"},{"supercategory": "animal","id": 25,"name": "giraffe"},{"supercategory": "accessory","id": 27,"name": "backpack"},{"supercategory": "accessory","id": 28,"name": "umbrella"},{"supercategory": "accessory","id": 31,"name": "handbag"},{"supercategory": "accessory","id": 32,"name": "tie"},{"supercategory": "accessory","id": 33,"name": "suitcase"},{"supercategory": "sports","id": 34,"name": "frisbee"},{"supercategory": "sports","id": 35,"name": "skis"},{"supercategory": "sports","id": 36,"name": "snowboard"},{"supercategory": "sports","id": 37,"name": "sports ball"},{"supercategory": "sports","id": 38,"name": "kite"},{"supercategory": "sports","id": 39,"name": "baseball bat"},{"supercategory": "sports","id": 40,"name": "baseball glove"},{"supercategory": "sports","id": 41,"name": "skateboard"},{"supercategory": "sports","id": 42,"name": "surfboard"},{"supercategory": "sports","id": 43,"name": "tennis racket"},{"supercategory": "kitchen","id": 44,"name": "bottle"},{"supercategory": "kitchen","id": 46,"name": "wine glass"},{"supercategory": "kitchen","id": 47,"name": "cup"},{"supercategory": "kitchen","id": 48,"name": "fork"},{"supercategory": "kitchen","id": 49,"name": "knife"},{"supercategory": "kitchen","id": 50,"name": "spoon"},{"supercategory": "kitchen","id": 51,"name": "bowl"},{"supercategory": "food","id": 52,"name": "banana"},{"supercategory": "food","id": 53,"name": "apple"},{"supercategory": "food","id": 54,"name": "sandwich"},{"supercategory": "food","id": 55,"name": "orange"},{"supercategory": "food","id": 56,"name": "broccoli"},{"supercategory": "food","id": 57,"name": "carrot"},{"supercategory": "food","id": 58,"name": "hot dog"},{"supercategory": "food","id": 59,"name": "pizza"},{"supercategory": "food","id": 60,"name": "donut"},{"supercategory": "food","id": 61,"name": "cake"},{"supercategory": "furniture","id": 62,"name": "chair"},{"supercategory": "furniture","id": 63,"name": "couch"},{"supercategory": "furniture","id": 64,"name": "potted plant"},{"supercategory": "furniture","id": 65,"name": "bed"},{"supercategory": "furniture","id": 67,"name": "dining table"},{"supercategory": "furniture","id": 70,"name": "toilet"},{"supercategory": "electronic","id": 72,"name": "tv"},{"supercategory": "electronic","id": 73,"name": "laptop"},{"supercategory": "electronic","id": 74,"name": "mouse"},{"supercategory": "electronic","id": 75,"name": "remote"},{"supercategory": "electronic","id": 76,"name": "keyboard"},{"supercategory": "electronic","id": 77,"name": "cell phone"},{"supercategory": "appliance","id": 78,"name": "microwave"},{"supercategory": "appliance","id": 79,"name": "oven"},{"supercategory": "appliance","id": 80,"name": "toaster"},{"supercategory": "appliance","id": 81,"name": "sink"},{"supercategory": "appliance","id": 82,"name": "refrigerator"},{"supercategory": "indoor","id": 84,"name": "book"},{"supercategory": "indoor","id": 85,"name": "clock"},{"supercategory": "indoor","id": 86,"name": "vase"},{"supercategory": "indoor","id": 87,"name": "scissors"},{"supercategory": "indoor","id": 88,"name": "teddy bear"},{"supercategory": "indoor","id": 89,"name": "hair drier"},{"supercategory": "indoor","id": 90,"name": "toothbrush"}],
```

Also, change the `category_id` in the `annotations` to the `id` corresponding to COCO, for example, `cat` is `17` in this example. Here are some of the results:

```json
  "annotations": [
    {
      "iscrowd": 0,
      "category_id": 17, # This "category_id" is changed to the id corresponding to COCO, for example, cat is 17
      "id": 32,
      "image_id": 32,
      "bbox": [
        822.49072265625,
        958.3897094726562,
        1513.693115234375,
        988.3231811523438
      ],
      "area": 1496017.9949368387,
      "segmentation": [
        [
          822.49072265625,
          958.3897094726562,
          822.49072265625,
          1946.712890625,
          2336.183837890625,
          1946.712890625,
          2336.183837890625,
          958.3897094726562
        ]
      ]
    }
  ]
```

3. executive command

```shell
python tools\test.py configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py \
                     work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth \
                     --cfg-options test_evaluator.classwise=True
```

After executing it, we can see the test metrics:

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
