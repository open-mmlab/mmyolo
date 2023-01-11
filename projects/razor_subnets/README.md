# Projecs Based on MMrazor

There are many research works and pretrained models built on MMrazor. We list some of them as examples of how to use MMrazor slimmable models for downstream framework. As the page might not be completed, please feel free to contribute more efficient models to update this page.

## Description

This is an implementation of MMrazor Searchable Backbone Application, we provide detection configs and models for MMrazor in MMyolo.

### Backbone support

#### NAS model
- [x] [AttentiveMobileNetV3](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/attentive_mobilenetv3_supernet.py)
- [x] [SearchableShuffleNetV2](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/spos_shufflenet_supernet.py)
- [x] [SearchableMobileNetV2](https://github.com/open-mmlab/mmrazor/blob/dev-1.x/configs/_base_/nas_backbones/spos_mobilenet_supernet.py)

## Usage
### Prerequisites
- [MMrazor v0.3.0](https://github.com/open-mmlab/mmrazor/tree/v0.3.0) or higher

### Training commands
In MMyolo's root directory, run the following command to train the model:
```bash
python tools/train.py projects/razor_subnets/configs/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py
```

For multi-gpu training, run:
```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=${NUM_GPUS} --master_port=29506 --master_addr="127.0.0.1" tools/train.py projects/razor_subnets/configs/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py
```

### Testing commands
In MMyolo's root directory, run the following command to test the model:
```bash
python tools/test.py projects/razor_subnets/configs/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py ${CHECKPOINT_PATH}
```


## Results and Models
Here we provide the baseline version of Yolo Series with NAS backbone.

|    Model    | size | box AP | Params(M) | FLOPS(G) |                      Config                        |                                                                                                                                                                 Download                                                                                                                                                                 |
| :---------: | :--: | :----: | :-------: | :------------------: | :-------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| yolov5_s_spos_shufflenetv2 | 640  |  37.9  |    7.04    |   7.03   | [config](./configs/yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117.log.json) |
| yolov6_l_attentive_a6   | 640  |  44.5  |   18.38   |  8.49   | [config](./configs/yolov6_l_attentivenas_a6_d12_syncbn_fast_16xb16-300e_coco.py) |       [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952.log.json)       |
| rtmdet_tiny_ofa_lat31   | 960  |  41.1  |   3.91    |   6.09   | [config](./configs/rtmdet_tiny_ofa_lat31_syncbn_16xb16-300e_coco.py) |       [model](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-0a8c901a.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329.log.json)       |

**Note**:

1. For a fair comparison, the config of training setting is consistent with the original model configuration, bringing about 0.2~0.5% AP improvement.

