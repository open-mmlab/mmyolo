# Projecs Based on MMRAZOR

There are many research works and pretrained models built on MMRAZOR. We list some of them as examples of how to use MMRAZOR slimmable models for downstream framework. As the page might not be completed, please feel free to contribute more efficient mmrazor-models to update this page.

## Description

This is an implementation of MMRAZOR Searchable Backbone Application, we provide detection configs and models for MMRAZOR in MMYOLO.

### Backbone support

#### NAS model

- \[x\] [AttentiveMobileNetV3](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/attentive_mobilenetv3_supernet.py)
- \[x\] [SearchableShuffleNetV2](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/spos_shufflenet_supernet.py)
- \[x\] [SearchableMobileNetV2](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/spos_mobilenet_supernet.py)

## Usage

### Prerequisites

- [MMRAZOR v1.0.0rc2](https://github.com/open-mmlab/mmrazor/tree/v1.0.0rc2) or higher (dev-1.x)

### Training commands

In MMYOLO's root directory, run the following command to train the model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh configs/razor_subnets/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py
```

### Testing commands

In MMYOLO's root directory, run the following command to test the model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_test.sh configs/razor_subnets/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py ${CHECKPOINT_PATH}
```

## Results and Models

Here we provide the baseline version of Yolo Series with NAS backbone.

|           Model            | size | box AP | Params(M) | FLOPS(G) |                                  Config                                  |                                                                           Download                                                                            |
| :------------------------: | :--: | :----: | :-------: | :------: | :----------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| yolov5_s_spos_shufflenetv2 | 640  |  37.9  |   7.04    |   7.03   |     [config](./yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py)     |         [model](https://download.openmmlab.com/mmrazor/v1/spos/yolov5/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco_20230109_155302-777fd6f1.pth)         |
|  yolov6_l_attentivenas_a6  | 640  |  44.5  |   18.38   |   8.49   | [config](./yolov6_l_attentivenas_a6_d12_syncbn_fast_16xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmrazor/v1/attentivenas/yolov6/yolov6_l_attentivenas_a6_d12_syncbn_fast_16xb16-300e_coco_20230108_174944-4970f0b7.pth) |
|   rtmdet_tiny_ofa_lat31    | 960  |  41.1  |   3.91    |   6.09   |       [config](./rtmdet_tiny_ofa_lat31_syncbn_16xb16-300e_coco.py)       |           [model](https://download.openmmlab.com/mmrazor/v1/ofa/rtmdet/rtmdet_tiny_ofa_lat31_syncbn_16xb16-300e_coco_20230108_222141-24ff87dex.pth)           |

**Note**:

1. For fair comparison, the training configuration is consistent with the original configuration and results in an improvement of about 0.2-0.5% AP.
1. `yolov5_s_spos_shufflenetv2` achieves 37.9% AP with only 7.042M parameters, directly instead of the backbone, and outperforms `yolov5_s` with a similar size by more than 0.2% AP.
1. With the efficient backbone of `yolov6_l_attentivenas_a6`, the input channels of `YOLOv6RepPAFPN` are reduced. Meanwhile, modify the **deepen_factor** and the neck is made deeper to restore the AP.
1. with the `rtmdet_tiny_ofa_lat31` backbone with only 3.315M parameters and 3.634G flops, we can modify the input resolution to 960, with a similar model size compared to `rtmdet_tiny` and exceeds `rtmdet_tiny` by 0.2% AP, reducing the size of the whole model to 3.91 MB.
