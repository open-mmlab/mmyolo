# Projecs Based on MMrazor

There are many research works and pretrained models built on MMrazor. We list some of them as examples of how to use MMrazor slimmable models for downstream framework. As the page might not be completed, please feel free to contribute more efficient models to update this page.

## Description

This is an implementation of MMrazor Searchable Backbone Application, we provide detection configs and models for MMrazor in MMyolo.

### Backbone support

#### NAS model

- \[x\] [AttentiveMobileNetV3](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/attentive_mobilenetv3_supernet.py)
- \[x\] [SearchableShuffleNetV2](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/spos_shufflenet_supernet.py)
- \[x\] [SearchableMobileNetV2](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/spos_mobilenet_supernet.py)

## Usage

### Prerequisites

- [MMrazor v1.0.0rc2](https://github.com/open-mmlab/mmrazor/tree/v1.0.0rc2) or higher (dev-1.x)

### Training commands

In MMyolo's root directory, run the following command to train the model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh configs/razor_subnets/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py
```

### Testing commands

In MMyolo's root directory, run the following command to test the model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_test.sh configs/razor_subnets/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py ${CHECKPOINT_PATH}
```

## Results and Models

Here we provide the baseline version of Yolo Series with NAS backbone.

|           Model            | size | box AP | Params(M) | FLOPS(G) |                                  Config                                  |                                                                                                                                                                                 Download                                                                                                                                                                                  |
|:--------------------------:|:----:|:------:|:---------:|:--------:|:------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| yolov5_s_spos_shufflenetv2 | 640  |  37.9  |   7.04    |   7.03   |     [config](./yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py)     |                 [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/spos/yolov5/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco_20230109_155302-777fd6f1.pth) | [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/spos/yolov5/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco_20230109_155302-777fd6f1.json)                 |
|  yolov6_l_attentivenas_a6  | 640  |  44.5  |   18.38   |   8.49   | [config](./yolov6_l_attentivenas_a6_d12_syncbn_fast_16xb16-300e_coco.py) | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/attentivenas/yolov6/yolov6_l_attentivenas_a6_d12_syncbn_fast_16xb16-300e_coco_20230108_174944-4970f0b7.pth) | [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/attentivenas/yolov6/yolov6_l_attentivenas_a6_d12_syncbn_fast_16xb16-300e_coco_20230108_174944-4970f0b7.json) |
|   rtmdet_tiny_ofa_lat31    | 960  |  41.1  |   3.91    |   6.09   |       [config](./rtmdet_tiny_ofa_lat31_syncbn_16xb16-300e_coco.py)       |                     [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/ofa/rtmdet/rtmdet_tiny_ofa_lat31_syncbn_16xb16-300e_coco_20230108_222141-24ff87dex.pth) | [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/ofa/rtmdet/rtmdet_tiny_ofa_lat31_syncbn_16xb16-300e_coco_20230108_222141-24ff87de.json)                      |

**Note**:

1. For a fair comparison, the config of training setting is consistent with the original model configuration, bringing about 0.2~0.5% AP improvement.
