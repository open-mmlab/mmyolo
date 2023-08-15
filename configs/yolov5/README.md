# YOLOv5

<!-- [ALGORITHM] -->

## Abstract

YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/200000324-70ae078f-cea7-4189-8baa-440656797dad.jpg"/>
YOLOv5-l-P5 model structure
</div>

<div align=center>
<img src="https://user-images.githubusercontent.com/27466624/211143533-1725c1b2-6189-4c3a-a046-ad968e03cb9d.jpg"/>
YOLOv5-l-P6 model structure
</div>

## Results and models

### COCO

| Backbone  | Arch | size | Mask Refine | SyncBN | AMP | Mem (GB) |   box AP    | TTA box AP |                                     Config                                      |                                                                                                                                                                                                       Download                                                                                                                                                                                                       |
| :-------: | :--: | :--: | :---------: | :----: | :-: | :------: | :---------: | :--------: | :-----------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOv5-n  |  P5  | 640  |     No      |  Yes   | Yes |   1.5    |    28.0     |    30.7    |             [config](./yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py)             |                                     [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739-b804c1ad.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco/yolov5_n-v61_syncbn_fast_8xb16-300e_coco_20220919_090739.log.json)                                     |
| YOLOv5-n  |  P5  | 640  |     Yes     |  Yes   | Yes |   1.5    |    28.0     |            | [config](./mask_refine/yolov5_n_mask-refine-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/mask_refine/yolov5_n_mask-refine-v61_syncbn_fast_8xb16-300e_coco/yolov5_n_mask-refine-v61_syncbn_fast_8xb16-300e_coco_20230305_152706-712fb1b2.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/mask_refine/yolov5_n_mask-refine-v61_syncbn_fast_8xb16-300e_coco/yolov5_n_mask-refine-v61_syncbn_fast_8xb16-300e_coco_20230305_152706.log.json) |
| YOLOv5u-n |  P5  | 640  |     Yes     |  Yes   | Yes |          |             |            | [config](./yolov5/yolov5u/yolov5_n_mask-refine_syncbn_fast_8xb16-300e_coco.py)  |                                                                                                                                                                                               [model](<>) \| [log](<>)                                                                                                                                                                                               |
| YOLOv5-s  |  P5  | 640  |     No      |  Yes   | Yes |   2.7    |    37.7     |    40.2    |             [config](./yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py)             |                                     [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700.log.json)                                     |
| YOLOv5-s  |  P5  | 640  |     Yes     |  Yes   | Yes |   2.7    | 38.0 (+0.3) |            | [config](./mask_refine/yolov5_s_mask-refine-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/mask_refine/yolov5_s_mask-refine-v61_syncbn_fast_8xb16-300e_coco/yolov5_s_mask-refine-v61_syncbn_fast_8xb16-300e_coco_20230304_033134-8e0cd271.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/mask_refine/yolov5_s_mask-refine-v61_syncbn_fast_8xb16-300e_coco/yolov5_s_mask-refine-v61_syncbn_fast_8xb16-300e_coco_20230304_033134.log.json) |
| YOLOv5u-s |  P5  | 640  |     Yes     |  Yes   | Yes |          |             |            | [config](./yolov5/yolov5u/yolov5_s_mask-refine_syncbn_fast_8xb16-300e_coco.py)  |                                                                                                                                                                                               [model](<>) \| [log](<>)                                                                                                                                                                                               |
| YOLOv5-m  |  P5  | 640  |     No      |  Yes   | Yes |   5.0    |    45.3     |    46.9    |             [config](./yolov5_m-v61_syncbn_fast_8xb16-300e_coco.py)             |                                     [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944.log.json)                                     |
| YOLOv5-m  |  P5  | 640  |     Yes     |  Yes   | Yes |   5.0    |    45.3     |            | [config](./mask_refine/yolov5_m_mask-refine-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/mask_refine/yolov5_m_mask-refine-v61_syncbn_fast_8xb16-300e_coco/yolov5_m_mask-refine-v61_syncbn_fast_8xb16-300e_coco_20230305_153946-44e96155.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/mask_refine/yolov5_m_mask-refine-v61_syncbn_fast_8xb16-300e_coco/yolov5_m_mask-refine-v61_syncbn_fast_8xb16-300e_coco_20230305_153946.log.json) |
| YOLOv5u-m |  P5  | 640  |     Yes     |  Yes   | Yes |          |             |            | [config](./yolov5/yolov5u/yolov5_m_mask-refine_syncbn_fast_8xb16-300e_coco.py)  |                                                                                                                                                                                               [model](<>) \| [log](<>)                                                                                                                                                                                               |
| YOLOv5-l  |  P5  | 640  |     No      |  Yes   | Yes |   8.1    |    48.8     |    49.9    |             [config](./yolov5_l-v61_syncbn_fast_8xb16-300e_coco.py)             |                                     [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_l-v61_syncbn_fast_8xb16-300e_coco/yolov5_l-v61_syncbn_fast_8xb16-300e_coco_20220917_031007-096ef0eb.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_l-v61_syncbn_fast_8xb16-300e_coco/yolov5_l-v61_syncbn_fast_8xb16-300e_coco_20220917_031007.log.json)                                     |
| YOLOv5-l  |  P5  | 640  |     Yes     |  Yes   | Yes |   8.1    | 49.3 (+0.5) |            | [config](./mask_refine/yolov5_l_mask-refine-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/mask_refine/yolov5_l_mask-refine-v61_syncbn_fast_8xb16-300e_coco/yolov5_l_mask-refine-v61_syncbn_fast_8xb16-300e_coco_20230305_154301-2c1d912a.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/mask_refine/yolov5_l_mask-refine-v61_syncbn_fast_8xb16-300e_coco/yolov5_l_mask-refine-v61_syncbn_fast_8xb16-300e_coco_20230305_154301.log.json) |
| YOLOv5u-l |  P5  | 640  |     Yes     |  Yes   | Yes |          |             |            | [config](./yolov5/yolov5u/yolov5_l_mask-refine_syncbn_fast_8xb16-300e_coco.py)  |                                                                                                                                                                                               [model](<>) \| [log](<>)                                                                                                                                                                                               |
| YOLOv5-x  |  P5  | 640  |     No      |  Yes   | Yes |   12.2   |    50.2     |            |             [config](./yolov5_x-v61_syncbn_fast_8xb16-300e_coco.py)             |                                     [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_x-v61_syncbn_fast_8xb16-300e_coco/yolov5_x-v61_syncbn_fast_8xb16-300e_coco_20230305_152943-00776a4b.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_x-v61_syncbn_fast_8xb16-300e_coco/yolov5_x-v61_syncbn_fast_8xb16-300e_coco_20230305_152943.log.json)                                     |
| YOLOv5-x  |  P5  | 640  |     Yes     |  Yes   | Yes |   12.2   | 50.9 (+0.7) |            | [config](./mask_refine/yolov5_x_mask-refine-v61_syncbn_fast_8xb16-300e_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/mask_refine/yolov5_x_mask-refine-v61_syncbn_fast_8xb16-300e_coco/yolov5_x_mask-refine-v61_syncbn_fast_8xb16-300e_coco_20230305_154321-07edeb62.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/mask_refine/yolov5_x_mask-refine-v61_syncbn_fast_8xb16-300e_coco/yolov5_x_mask-refine-v61_syncbn_fast_8xb16-300e_coco_20230305_154321.log.json) |
| YOLOv5u-x |  P5  | 640  |     Yes     |  Yes   | Yes |          |             |            | [config](./yolov5/yolov5u/yolov5_x_mask-refine_syncbn_fast_8xb16-300e_coco.py)  |                                                                                                                                                                                               [model](<>) \| [log](<>)                                                                                                                                                                                               |
| YOLOv5-n  |  P6  | 1280 |     No      |  Yes   | Yes |   5.8    |    35.9     |            |           [config](./yolov5_n-p6-v62_syncbn_fast_8xb16-300e_coco.py)            |                               [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-p6-v62_syncbn_fast_8xb16-300e_coco/yolov5_n-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_224705-d493c5f3.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-p6-v62_syncbn_fast_8xb16-300e_coco/yolov5_n-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_224705.log.json)                               |
| YOLOv5-s  |  P6  | 1280 |     No      |  Yes   | Yes |   10.5   |    44.4     |            |           [config](./yolov5_s-p6-v62_syncbn_fast_8xb16-300e_coco.py)            |                               [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-p6-v62_syncbn_fast_8xb16-300e_coco/yolov5_s-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_215044-58865c19.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-p6-v62_syncbn_fast_8xb16-300e_coco/yolov5_s-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_215044.log.json)                               |
| YOLOv5-m  |  P6  | 1280 |     No      |  Yes   | Yes |   19.1   |    51.3     |            |           [config](./yolov5_m-p6-v62_syncbn_fast_8xb16-300e_coco.py)            |                               [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-p6-v62_syncbn_fast_8xb16-300e_coco/yolov5_m-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_230453-49564d58.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-p6-v62_syncbn_fast_8xb16-300e_coco/yolov5_m-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_230453.log.json)                               |
| YOLOv5-l  |  P6  | 1280 |     No      |  Yes   | Yes |   30.5   |    53.7     |            |           [config](./yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco.py)            |                               [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_234308-7a2ba6bf.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_234308.log.json)                               |

**Note**:

1. `fast` means that `YOLOv5DetDataPreprocessor` and `yolov5_collate` are used for data preprocessing, which is faster for training, but less flexible for multitasking. Recommended to use fast version config if you only care about object detection.
2. `detect` means that the network input is fixed to `640x640` and the post-processing thresholds is modified.
3. `SyncBN` means use SyncBN, `AMP` indicates training with mixed precision.
4. We use 8x A100 for training, and the single-GPU batch size is 16. This is different from the official code.
5. The performance is unstable and may fluctuate by about 0.4 mAP and the highest performance weight in `COCO` training in `YOLOv5` may not be the last epoch.
6. `TTA` means that Test Time Augmentation. It's perform 3 multi-scaling transformations on the image, followed by 2 flipping transformations (flipping and not flipping). You only need to specify `--tta` when testing to enable.  see [TTA](https://github.com/open-mmlab/mmyolo/blob/dev/docs/en/common_usage/tta.md) for details.
7. The performance of `Mask Refine` training is for the weight performance officially released by YOLOv5. `Mask Refine` means refining bbox by mask while loading annotations and transforming after `YOLOv5RandomAffine`, `Copy Paste` means using `YOLOv5CopyPaste`.
8. `YOLOv5u` models use the same loss functions and split Detect head as `YOLOv8` models for improved performance, but only requires 300 epochs.

### COCO Instance segmentation

|       Backbone        | Arch | size | SyncBN | AMP | Mem (GB) | Box AP | Mask AP |                                          Config                                          |                                                                                                                                                                                                                             Download                                                                                                                                                                                                                             |
| :-------------------: | :--: | :--: | :----: | :-: | :------: | :----: | :-----: | :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       YOLOv5-n        |  P5  | 640  |  Yes   | Yes |   3.3    |  27.9  |  23.7   |       [config](./ins_seg/yolov5_ins_n-v61_syncbn_fast_8xb16-300e_coco_instance.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_n-v61_syncbn_fast_8xb16-300e_coco_instance/yolov5_ins_n-v61_syncbn_fast_8xb16-300e_coco_instance_20230424_104807-84cc9240.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_n-v61_syncbn_fast_8xb16-300e_coco_instance/yolov5_ins_n-v61_syncbn_fast_8xb16-300e_coco_instance_20230424_104807.log.json)                         |
|       YOLOv5-s        |  P5  | 640  |  Yes   | Yes |   4.8    |  38.1  |  32.0   |       [config](./ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance_20230426_012542-3e570436.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance_20230426_012542.log.json)                         |
| YOLOv5-s(non-overlap) |  P5  | 640  |  Yes   | Yes |   4.8    |  38.0  |  32.1   | [config](./ins_seg/yolov5_ins_s-v61_syncbn_fast_non_overlap_8xb16-300e_coco_instance.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_non_overlap_8xb16-300e_coco_instance/yolov5_ins_s-v61_syncbn_fast_non_overlap_8xb16-300e_coco_instance_20230424_104642-6780d34e.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_s-v61_syncbn_fast_non_overlap_8xb16-300e_coco_instance/yolov5_ins_s-v61_syncbn_fast_non_overlap_8xb16-300e_coco_instance_20230424_104642.log.json) |
|       YOLOv5-m        |  P5  | 640  |  Yes   | Yes |   7.3    |  45.1  |  37.3   |       [config](./ins_seg/yolov5_ins_m-v61_syncbn_fast_8xb16-300e_coco_instance.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_m-v61_syncbn_fast_8xb16-300e_coco_instance/yolov5_ins_m-v61_syncbn_fast_8xb16-300e_coco_instance_20230424_111529-ef5ba1a9.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_m-v61_syncbn_fast_8xb16-300e_coco_instance/yolov5_ins_m-v61_syncbn_fast_8xb16-300e_coco_instance_20230424_111529.log.json)                         |
|       YOLOv5-l        |  P5  | 640  |  Yes   | Yes |   10.7   |  48.8  |  39.9   |       [config](./ins_seg/yolov5_ins_l-v61_syncbn_fast_8xb16-300e_coco_instance.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_l-v61_syncbn_fast_8xb16-300e_coco_instance/yolov5_ins_l-v61_syncbn_fast_8xb16-300e_coco_instance_20230508_104049-daa09f70.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_l-v61_syncbn_fast_8xb16-300e_coco_instance/yolov5_ins_l-v61_syncbn_fast_8xb16-300e_coco_instance_20230508_104049.log.json)                         |
|       YOLOv5-x        |  P5  | 640  |  Yes   | Yes |   15.0   |  50.6  |  41.4   |       [config](./ins_seg/yolov5_ins_x-v61_syncbn_fast_8xb16-300e_coco_instance.py)       |                         [model](https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_x-v61_syncbn_fast_8xb16-300e_coco_instance/yolov5_ins_x-v61_syncbn_fast_8xb16-300e_coco_instance_20230508_103925-a260c798.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/ins_seg/yolov5_ins_x-v61_syncbn_fast_8xb16-300e_coco_instance/yolov5_ins_x-v61_syncbn_fast_8xb16-300e_coco_instance_20230508_103925.log.json)                         |

**Note**:

1. `Non-overlap` refers to the instance-level masks being stored in the format (num_instances, h, w) instead of (h, w). Storing masks in overlap format consumes less memory and GPU memory.
2. For the M model, the `affine_scale` parameter should be 0.9, but due to some reason, we set it to 0.5 and found that the mAP did not change. Therefore, the released M model has an `affine_scale` parameter of 0.5, which is inconsistent with the value of 0.9 in the configuration.

### VOC

| Backbone | size | Batchsize | AMP | Mem (GB) | box AP(COCO metric) |                          Config                           |                                                                                                                                                 Download                                                                                                                                                 |
| :------: | :--: | :-------: | :-: | :------: | :-----------------: | :-------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOv5-n | 512  |    64     | Yes |   3.5    |        51.2         | [config](./yolov5/voc/yolov5_n-v61_fast_1xb64-50e_voc.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_fast_1xb64-50e_voc/yolov5_n-v61_fast_1xb64-50e_voc_20221017_234254-f1493430.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_n-v61_fast_1xb64-50e_voc/yolov5_n-v61_fast_1xb64-50e_voc_20221017_234254.log.json) |
| YOLOv5-s | 512  |    64     | Yes |   6.5    |        62.7         | [config](./yolov5/voc/yolov5_s-v61_fast_1xb64-50e_voc.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_fast_1xb64-50e_voc/yolov5_s-v61_fast_1xb64-50e_voc_20221017_234156-0009b33e.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_fast_1xb64-50e_voc/yolov5_s-v61_fast_1xb64-50e_voc_20221017_234156.log.json) |
| YOLOv5-m | 512  |    64     | Yes |   12.0   |        70.1         | [config](./yolov5/voc/yolov5_m-v61_fast_1xb64-50e_voc.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_fast_1xb64-50e_voc/yolov5_m-v61_fast_1xb64-50e_voc_20221017_114138-815c143a.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_fast_1xb64-50e_voc/yolov5_m-v61_fast_1xb64-50e_voc_20221017_114138.log.json) |
| YOLOv5-l | 512  |    32     | Yes |   10.0   |        73.1         | [config](./yolov5/voc/yolov5_l-v61_fast_1xb32-50e_voc.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_l-v61_fast_1xb32-50e_voc/yolov5_l-v61_fast_1xb32-50e_voc_20221017_045500-edc7e0d8.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_l-v61_fast_1xb32-50e_voc/yolov5_l-v61_fast_1xb32-50e_voc_20221017_045500.log.json) |

**Note**:

1. Training on VOC dataset need pretrained model which trained on COCO.
2. The performance is unstable and may fluctuate by about 0.4 mAP.
3. Official YOLOv5 use COCO metric, while training VOC dataset.
4. We converted the VOC test dataset to COCO format offline, while reproducing mAP result as shown above. We will support to use COCO metric while training VOC dataset in later version.
5. Hyperparameter reference from `https://wandb.ai/glenn-jocher/YOLOv5_VOC_official`.

### CrowdHuman

Since the `iscrowd` annotation of the COCO dataset is not equivalent to `ignore`, we use the CrowdHuman dataset to verify that the YOLOv5 ignore logic is correct.

| Backbone | size | SyncBN | AMP | Mem (GB) | ignore_iof_thr | box AP50(CrowDHuman Metric) |  MR  |  JI   |                                   Config                                   | Download |
| :------: | :--: | :----: | :-: | :------: | :------------: | :-------------------------: | :--: | :---: | :------------------------------------------------------------------------: | :------: |
| YOLOv5-s | 640  |  Yes   | Yes |   2.6    |       -1       |            85.79            | 48.7 | 75.33 |  [config](./yolov5/crowdhuman/yolov5_s-v61_fast_8xb16-300e_crowdhuman.py)  |          |
| YOLOv5-s | 640  |  Yes   | Yes |   2.6    |      0.5       |            86.17            | 48.8 | 75.87 | [config](./yolov5/crowdhuman/yolov5_s-v61_8xb16-300e_ignore_crowdhuman.py) |          |

**Note**:

1. `ignore_iof_thr` is -1 indicating that the ignore tag is not considered. We adjusted with `ignore_iof_thr` thresholds of 0.5, 0.8, 0.9, and the results show that 0.5 has the best performance.
2. The above table shows the performance of the model with the best performance on the validation set. The best performing models are around 160+ epoch which means that there is no need to train so many epochs.
3. This is a very simple implementation that simply replaces COCO's anchor with the `tools/analysis_tools/optimize_anchors.py` script. We'll adjust other parameters later to improve performance.

## Citation

```latex
@software{glenn_jocher_2022_7002879,
  author       = {Glenn Jocher and
                  Ayush Chaurasia and
                  Alex Stoken and
                  Jirka Borovec and
                  NanoCode012 and
                  Yonghye Kwon and
                  TaoXie and
                  Kalen Michael and
                  Jiacong Fang and
                  imyhxy and
                  Lorna and
                  Colin Wong and
                  曾逸夫(Zeng Yifu) and
                  Abhiram V and
                  Diego Montes and
                  Zhiqiang Wang and
                  Cristi Fati and
                  Jebastin Nadar and
                  Laughing and
                  UnglvKitDe and
                  tkianai and
                  yxNONG and
                  Piotr Skalski and
                  Adam Hogan and
                  Max Strobel and
                  Mrinal Jain and
                  Lorenzo Mammana and
                  xylieong},
  title        = {{ultralytics/yolov5: v6.2 - YOLOv5 Classification
                   Models, Apple M1, Reproducibility, ClearML and
                   Deci.ai integrations}},
  month        = aug,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v6.2},
  doi          = {10.5281/zenodo.7002879},
  url          = {https://doi.org/10.5281/zenodo.7002879}
}
```
