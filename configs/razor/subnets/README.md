# Projecs Based on MMRazor

There are many research works and pre-trained models built on MMRazor. We list some of them as examples of how to use MMRazor slimmable models for downstream frameworks. As the page might not be completed, please feel free to contribute more efficient mmrazor-models to update this page.

## Description

This is an implementation of MMRazor Searchable Backbone Application, we provide detection configs and models for MMRazor in MMYOLO.

### Backbone support

Here are the Neural Architecture Search(NAS) Models that come from MMRazor which support YOLO Series. If you are looking for MMRazor models only for Backbone, you could refer to MMRazor [ModelZoo](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/docs/en/get_started/model_zoo.md) and corresponding repository.

- [x] [AttentiveMobileNetV3](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/attentive_mobilenetv3_supernet.py)
- [x] [SearchableShuffleNetV2](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/spos_shufflenet_supernet.py)
- [x] [SearchableMobileNetV2](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/spos_mobilenet_supernet.py)

## Usage

### Prerequisites

- [MMRazor v1.0.0rc2](https://github.com/open-mmlab/mmrazor/tree/v1.0.0rc2) or higher (dev-1.x)

Install MMRazor using MIM.

```shell
mim install mmengine
mim install "mmrazor>=1.0.0rc2"
```

Install MMRazor from source

```
git clone -b dev-1.x https://github.com/open-mmlab/mmrazor.git
cd mmrazor
# Install MMRazor
mim install -v -e .
```

### Training commands

In MMYOLO's root directory, if you want to use single GPU for training, run the following command to train the model:

```bash
CUDA_VISIBLE_DEVICES=0 PORT=29500 ./tools/dist_train.sh configs/razor/subnets/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py
```

If you want to use several of these GPUs to train in parallel, you can use the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh configs/razor/subnets/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py
```

### Testing commands

In MMYOLO's root directory, run the following command to test the model:

```bash
CUDA_VISIBLE_DEVICES=0 PORT=29500 ./tools/dist_test.sh configs/razor/subnets/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py ${CHECKPOINT_PATH}
```

## Results and Models

Here we provide the baseline version of YOLO Series with NAS backbone.

|           Model            | size | box AP |  Params(M)   | FLOPs(G) |                                 Config                                  |                                                                                                                                                                   Download                                                                                                                                                                   |
| :------------------------: | :--: | :----: | :----------: | :------: | :---------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|          yolov5-s          | 640  |  37.7  |    7.235     |  8.265   |   [config](../../yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py)    | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json) |
| yolov5_s_spos_shufflenetv2 | 640  |  38.0  | 7.04(-2.7%)  |   7.03   |    [config](./yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py)     |                                                                                          [model](https://download.openmmlab.com/mmrazor/v1/yolo_nas_backbone/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco_20230211_220635-578be9a9.pth) \| log                                                                                          |
|          yolov6-s          | 640  |  44.0  |    18.869    |  24.253  |     [config](../../yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py)      |         [model](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035.log.json)         |
|  yolov6_l_attentivenas_a6  | 640  |  45.3  | 18.38(-2.6%) |   8.49   | [config](./yolov6_l_attentivenas_a6_d12_syncbn_fast_8xb32-300e_coco.py) |                                                                                      [model](https://download.openmmlab.com/mmrazor/v1/yolo_nas_backbone/yolov6_l_attentivenas_a6_d12_syncbn_fast_8xb32-300e_coco_20230211_222409-dcc72668.pth) \| log                                                                                       |
|        RTMDet-tiny         | 640  |  41.0  |     4.8      |   8.1    |     [config](../../rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py)      |   [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117.log.json)   |
|   rtmdet_tiny_ofa_lat31    | 960  |  41.3  | 3.91(-18.5%) |   6.09   |      [config](./rtmdet_tiny_ofa_lat31_syncbn_16xb16-300e_coco.py)       |                                                                                            [model](https://download.openmmlab.com/mmrazor/v1/yolo_nas_backbone/rtmdet_tiny_ofa_lat31_syncbn_16xb16-300e_coco_20230214_210623-449bb2a0.pth) \| log                                                                                            |

**Note**:

1. For fair comparison, the training configuration is consistent with the original configuration and results in an improvement of about 0.2-0.5% AP.
2. `yolov5_s_spos_shufflenetv2` achieves 38.0% AP with only 7.042M parameters, directly instead of the backbone, and outperforms `yolov5_s` with a similar size by more than 0.3% AP.
3. With the efficient backbone of `yolov6_l_attentivenas_a6`, the input channels of `YOLOv6RepPAFPN` are reduced. Meanwhile, modify the **deepen_factor** and the neck is made deeper to restore the AP.
4. with the `rtmdet_tiny_ofa_lat31` backbone with only 3.315M parameters and 3.634G flops, we can modify the input resolution to 960, with a similar model size compared to `rtmdet_tiny` and exceeds `rtmdet_tiny` by 0.4% AP, reducing the size of the whole model to 3.91 MB.
