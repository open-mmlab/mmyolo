# How to xxx

This tutorial collects answers to any `How to xxx with MMYOLO`. Feel free to update this doc if you meet new questions about `How to` and find the answers!

## Add plugins to the backbone network

Please see [Plugins](plugins.md).

## Apply multiple Necks

If you want to stack multiple Necks, you can directly set the Neck parameters in the config. MMYOLO supports concatenating multiple Necks in the form of `List`. You need to ensure that the output channel of the previous Neck matches the input channel of the next Neck. If you need to adjust the number of channels, you can insert the `mmdet.ChannelMapper` module to align the number of channels between multiple Necks. The specific configuration is as follows:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

deepen_factor = _base_.deepen_factor
widen_factor = _base_.widen_factor
model = dict(
    type='YOLODetector',
    neck=[
        dict(
            type='YOLOv5PAFPN',
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            in_channels=[256, 512, 1024],
            out_channels=[256, 512, 1024], # The out_channels is controlled by widen_factor，so the YOLOv5PAFPN's out_channels equls to out_channels * widen_factor
            num_csp_blocks=3,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True)),
        dict(
            type='mmdet.ChannelMapper',
            in_channels=[128, 256, 512],
            out_channels=128,
        ),
        dict(
            type='mmdet.DyHead',
            in_channels=128,
            out_channels=256,
            num_blocks=2,
            # disable zero_init_offset to follow official implementation
            zero_init_offset=False)
    ]
    bbox_head=dict(head_module=dict(in_channels=[512,512,512])) # The out_channels is controlled by widen_factor，so the YOLOv5HeadModuled in_channels * widen_factor equals to  the last neck's out_channels
)
```

## Replace the backbone network

```{note}
1. When using other backbone networks, you need to ensure that the output channels of the backbone network match the input channels of the neck network.
2. The configuration files given below only ensure that the training will work correctly, and their training performance may not be optimal. Because some backbones require specific learning rates, optimizers, and other hyperparameters. Related contents will be added in the "Training Tips" section later.
```

### Use backbone network implemented in MMYOLO

Suppose you want to use `YOLOv6EfficientRep` as the backbone network of `YOLOv5`, the example config is as the following:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    backbone=dict(
        type='YOLOv6EfficientRep',
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True))
)
```

### Use backbone network implemented in other OpenMMLab repositories

The model registry in MMYOLO, MMDetection, MMClassification, and MMSegmentation all inherit from the root registry in MMEngine in the OpenMMLab 2.0 system, allowing these repositories to directly use modules already implemented by each other. Therefore, in MMYOLO, users can use backbone networks from MMDetection and MMClassification without reimplementation.

#### Use backbone network implemented in MMDetection

1. Suppose you want to use `ResNet-50` as the backbone network of `YOLOv5`, the example config is as the following:

   ```python
   _base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

   deepen_factor = _base_.deepen_factor
   widen_factor = 1.0
   channels = [512, 1024, 2048]

   model = dict(
       backbone=dict(
           _delete_=True, # Delete the backbone field in _base_
           type='mmdet.ResNet', # Using ResNet from mmdet
           depth=50,
           num_stages=4,
           out_indices=(1, 2, 3),
           frozen_stages=1,
           norm_cfg=dict(type='BN', requires_grad=True),
           norm_eval=True,
           style='pytorch',
           init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
       neck=dict(
           type='YOLOv5PAFPN',
           widen_factor=widen_factor,
           in_channels=channels, # Note: The 3 channels of ResNet-50 output are [512, 1024, 2048], which do not match the original yolov5-s neck and need to be changed.
           out_channels=channels),
       bbox_head=dict(
           type='YOLOv5Head',
           head_module=dict(
               type='YOLOv5HeadModule',
               in_channels=channels, # input channels of head need to be changed accordingly
               widen_factor=widen_factor))
   )
   ```

2. Suppose you want to use `SwinTransformer-Tiny` as the backbone network of `YOLOv5`, the example config is as the following:

   ```python
   _base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

   deepen_factor = _base_.deepen_factor
   widen_factor = 1.0
   channels = [192, 384, 768]
   checkpoint_file = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

   model = dict(
       backbone=dict(
           _delete_=True, # Delete the backbone field in _base_
           type='mmdet.SwinTransformer', # Using SwinTransformer from mmdet
           embed_dims=96,
           depths=[2, 2, 6, 2],
           num_heads=[3, 6, 12, 24],
           window_size=7,
           mlp_ratio=4,
           qkv_bias=True,
           qk_scale=None,
           drop_rate=0.,
           attn_drop_rate=0.,
           drop_path_rate=0.2,
           patch_norm=True,
           out_indices=(1, 2, 3),
           with_cp=False,
           convert_weights=True,
           init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
       neck=dict(
           type='YOLOv5PAFPN',
           deepen_factor=deepen_factor,
           widen_factor=widen_factor,
           in_channels=channels, # Note: The 3 channels of SwinTransformer-Tiny output are [192, 384, 768], which do not match the original yolov5-s neck and need to be changed.
           out_channels=channels),
       bbox_head=dict(
           type='YOLOv5Head',
           head_module=dict(
               type='YOLOv5HeadModule',
               in_channels=channels, # input channels of head need to be changed accordingly
               widen_factor=widen_factor))
   )
   ```

#### Use backbone network implemented in MMClassification

1. Suppose you want to use `ConvNeXt-Tiny` as the backbone network of `YOLOv5`, the example config is as the following:

   ```python
   _base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

   # please run the command, mim install "mmcls>=1.0.0rc2", to install mmcls
   # import mmcls.models to trigger register_module in mmcls
   custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
   checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa
   deepen_factor = _base_.deepen_factor
   widen_factor = 1.0
   channels = [192, 384, 768]

   model = dict(
       backbone=dict(
           _delete_=True, # Delete the backbone field in _base_
           type='mmcls.ConvNeXt', # Using ConvNeXt from mmcls
           arch='tiny',
           out_indices=(1, 2, 3),
           drop_path_rate=0.4,
           layer_scale_init_value=1.0,
           gap_before_final_norm=False,
           init_cfg=dict(
               type='Pretrained', checkpoint=checkpoint_file,
               prefix='backbone.')), # The pre-trained weights of backbone network in MMCls have prefix='backbone.'. The prefix in the keys will be removed so that these weights can be normally loaded.
       neck=dict(
           type='YOLOv5PAFPN',
           deepen_factor=deepen_factor,
           widen_factor=widen_factor,
           in_channels=channels, # Note: The 3 channels of ConvNeXt-Tiny output are [192, 384, 768], which do not match the original yolov5-s neck and need to be changed.
           out_channels=channels),
       bbox_head=dict(
           type='YOLOv5Head',
           head_module=dict(
               type='YOLOv5HeadModule',
               in_channels=channels, # input channels of head need to be changed accordingly
               widen_factor=widen_factor))
   )
   ```

2. Suppose you want to use `MobileNetV3-small` as the backbone network of `YOLOv5`, the example config is as the following:

   ```python
   _base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

   # please run the command, mim install "mmcls>=1.0.0rc2", to install mmcls
   # import mmcls.models to trigger register_module in mmcls
   custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
   checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_small-8427ecf0.pth'  # noqa
   deepen_factor = _base_.deepen_factor
   widen_factor = 1.0
   channels = [24, 48, 96]

   model = dict(
       backbone=dict(
           _delete_=True, # Delete the backbone field in _base_
           type='mmcls.MobileNetV3', # Using MobileNetV3 from mmcls
           arch='small',
           out_indices=(3, 8, 11), # Modify out_indices
           init_cfg=dict(
               type='Pretrained',
               checkpoint=checkpoint_file,
               prefix='backbone.')), # The pre-trained weights of backbone network in MMCls have prefix='backbone.'. The prefix in the keys will be removed so that these weights can be normally loaded.
       neck=dict(
           type='YOLOv5PAFPN',
           deepen_factor=deepen_factor,
           widen_factor=widen_factor,
           in_channels=channels, # Note: The 3 channels of MobileNetV3 output are [24, 48, 96], which do not match the original yolov5-s neck and need to be changed.
           out_channels=channels),
       bbox_head=dict(
           type='YOLOv5Head',
           head_module=dict(
               type='YOLOv5HeadModule',
               in_channels=channels, # input channels of head need to be changed accordingly
               widen_factor=widen_factor))
   )
   ```

#### Use backbone network in `timm` through MMClassification

MMClassification also provides a wrapper for the Py**T**orch **Im**age **M**odels (`timm`) backbone network, users can directly use the backbone network in `timm` through MMClassification. Suppose you want to use `EfficientNet-B1` as the backbone network of `YOLOv5`, the example config is as the following:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# please run the command, mim install "mmcls>=1.0.0rc2", to install mmcls
# and the command, pip install timm, to install timm
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)

deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [40, 112, 320]

model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmcls.TIMMBackbone', # Using timm from mmcls
        model_name='efficientnet_b1', # Using efficientnet_b1 in timm
        features_only=True,
        pretrained=True,
        out_indices=(2, 3, 4)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=channels, # Note: The 3 channels of EfficientNet-B1 output are [40, 112, 320], which do not match the original yolov5-s neck and need to be changed.
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels, # input channels of head need to be changed accordingly
            widen_factor=widen_factor))
)
```

#### Use backbone network implemented in MMSelfSup

Suppose you want to use `ResNet-50` which is self-supervised trained by `MoCo v3` in MMSelfSup as the backbone network of `YOLOv5`, the example config is as the following:

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# please run the command, mim install "mmselfsup>=1.0.0rc3", to install mmselfsup
# import mmselfsup.models to trigger register_module in mmselfsup
custom_imports = dict(imports=['mmselfsup.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220927-e043f51a.pth'  # noqa
deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [512, 1024, 2048]

model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmselfsup.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3, 4), # Note: out_indices of ResNet in MMSelfSup are 1 larger than those in MMdet and MMCls
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=channels, # Note: The 3 channels of ResNet-50 output are [512, 1024, 2048], which do not match the original yolov5-s neck and need to be changed.
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels, # input channels of head need to be changed accordingly
            widen_factor=widen_factor))
)
```

#### Don't used pre-training weights

When we replace the backbone network, the model initialization is trained by default loading the pre-training weight of the backbone network. Instead of using the pre-training weights of the backbone network, if you want to train the time model from scratch,
You can set `init_cfg` in 'backbone' to 'None'. In this case, the backbone network will be initialized with the default initialization method, instead of using the trained pre-training weight.

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

deepen_factor = _base_.deepen_factor
widen_factor = 1.0
channels = [512, 1024, 2048]

model = dict(
   backbone=dict(
       _delete_=True, # Delete the backbone field in _base_
       type='mmdet.ResNet', # Using ResNet from mmdet
       depth=50,
       num_stages=4,
       out_indices=(1, 2, 3),
       frozen_stages=1,
       norm_cfg=dict(type='BN', requires_grad=True),
       norm_eval=True,
       style='pytorch',
       init_cfg=None # If init_cfg is set to None, backbone will not be initialized with pre-trained weights
   ),
   neck=dict(
       type='YOLOv5PAFPN',
       widen_factor=widen_factor,
       in_channels=channels, # Note: The 3 channels of ResNet-50 output are [512, 1024, 2048], which do not match the original yolov5-s neck and need to be changed.
       out_channels=channels),
   bbox_head=dict(
       type='YOLOv5Head',
       head_module=dict(
           type='YOLOv5HeadModule',
           in_channels=channels, # input channels of head need to be changed accordingly
           widen_factor=widen_factor))
)
```

#### Freeze the weight of backbone or neck

In MMYOLO, we can freeze some `stages` of the backbone network by setting `frozen_stages` parameters, so that these `stage` parameters do not participate in model updating.
It should be noted that `frozen_stages = i` means that all parameters from the initial `stage` to the `i`<sup>th</sup> `stage` will be frozen. The following is an example of `YOLOv5`. Other algorithms are the same logic.

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    backbone=dict(
        frozen_stages=1 # Indicates that the parameters in the first stage and all stages before it are frozen
    ))
```

In addition, it's able to freeze the whole `neck` with the parameter `freeze_all` in MMYOLO. The following is an example of `YOLOv5`. Other algorithms are the same logic.

```python
_base_ = './yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    neck=dict(
        freeze_all=True # If freeze_all=True, all parameters of the neck will be frozen
    ))
```

## Output prediction results

If you want to save the prediction results as a specific file for offline evaluation, MMYOLO currently supports both json and pkl formats.

```{note}
The json file only save `image_id`, `bbox`, `score` and `category_id`. The json file can be read using the json library.
The pkl file holds more content than the json file, and also holds information such as the file name and size of the predicted image; the pkl file can be read using the pickle library. The pkl file can be read using the pickle library.
```

### Output into json file

If you want to output the prediction results as a json file, the command is as follows.

```shell
python tools/test.py {path_to_config} {path_to_checkpoint} --json-prefix {json_prefix}
```

The argument after `--json-prefix` should be a filename prefix (no need to enter the `.json` suffix) and can also contain a path. For a concrete example:

```shell
python tools/test.py configs\yolov5\yolov5_s-v61_syncbn_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --json-prefix work_dirs/demo/json_demo
```

Running the above command will output the `json_demo.bbox.json` file in the `work_dirs/demo` folder.

### Output into pkl file

If you want to output the prediction results as a pkl file, the command is as follows.

```shell
python tools/test.py {path_to_config} {path_to_checkpoint} --out {path_to_output_file}
```

The argument after `--out` should be a full filename (**must be** with a `.pkl` or `.pickle` suffix) and can also contain a path. For a concrete example:

```shell
python tools/test.py configs\yolov5\yolov5_s-v61_syncbn_8xb16-300e_coco.py yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth --out work_dirs/demo/pkl_demo.pkl
```

Running the above command will output the `pkl_demo.pkl` file in the `work_dirs/demo` folder.

## Use mim to run scripts from other OpenMMLab repositories

```{note}
1. All script calls across libraries are currently not supported and are being fixed. More examples will be added to this document when the fix is complete. 2.
2. mAP plotting and average training speed calculation are fixed in the MMDetection dev-3.x branch, which currently needs to be installed via the source code to be run successfully.
```

## Log Analysis

#### Curve plotting

`tools/analysis_tools/analyze_logs.py` plots loss/mAP curves given a training log file. Run `pip install seaborn` first to install the dependency.

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

#### Compute the average training speed

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

### Print the whole config

`print_config.py` in MMDetection prints the whole config verbatim, expanding all its imports. The command is as following.

```shell
mim run mmdet print_config \
    ${CONFIG} \                              # path of the config file
    [--save-path] \                          # save path of whole config, suffixed with .py, .json or .yml
    [--cfg-options ${OPTIONS [OPTIONS...]}]  # override some settings in the used config
```

Examples:

```shell
mim run mmdet print_config \
    configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py \
    --save-path ./work_dirs/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py
```

Running the above command will save the `yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py` config file with the inheritance relationship expanded to \`\`yolov5_s-v61_syncbn_fast_1xb4-300e_balloon_whole.py`in the`./work_dirs\` folder.

## Set the random seed

If you want to set the random seed during training, you can use the following command.

```shell
python ./tools/train.py \
    ${CONFIG} \                               # path of the config file
    --cfg-options randomness.seed=2023 \      # set seed to 2023
    [randomness.diff_rank_seed=True] \        # set different seeds according to global rank
    [randomness.deterministic=True]           # set the deterministic option for CUDNN backend
# [] stands for optional parameters, when actually entering the command line, you do not need to enter []
```

`randomness` has three parameters that can be set, with the following meanings.

- `randomness.seed=2023`, set the random seed to 2023.
- `randomness.diff_rank_seed=True`, set different seeds according to global rank. Defaults to False.
- `randomness.deterministic=True`, set the deterministic option for cuDNN backend, i.e., set `torch.backends.cudnn.deterministic` to True and `torch.backends.cudnn.benchmark` to False. Defaults to False. See https://pytorch.org/docs/stable/notes/randomness.html for more details.

## Specify specific GPUs during training or inference

If you have multiple GPUs, such as 8 GPUs, numbered `0, 1, 2, 3, 4, 5, 6, 7`, GPU 0 will be used by default for training or inference. If you want to specify other GPUs for training or inference, you can use the following commands:

```shell
CUDA_VISIBLE_DEVICES=5 python ./tools/train.py ${CONFIG} #train
CUDA_VISIBLE_DEVICES=5 python ./tools/test.py ${CONFIG} ${CHECKPOINT_FILE} #test
```

If you set `CUDA_VISIBLE_DEVICES` to -1 or a number greater than the maximum GPU number, such as 8, the CPU will be used for training or inference.

If you want to use several of these GPUs to train in parallel, you can use the following command:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ${CONFIG} ${GPU_NUM}
```

Here the `GPU_NUM` is 4. In addition, if multiple tasks are trained in parallel on one machine and each task requires multiple GPUs, the PORT of each task need to be set differently to avoid communication conflict, like the following commands:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG} 4
```
