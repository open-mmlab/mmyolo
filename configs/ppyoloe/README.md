# PPYOLOE

<!-- [ALGORITHM] -->

## Abstract

PP-YOLOE is an excellent single-stage anchor-free model based on PP-YOLOv2, surpassing a variety of popular YOLO models. PP-YOLOE has a series of models, named s/m/l/x, which are configured through width multiplier and depth multiplier. PP-YOLOE avoids using special operators, such as Deformable Convolution or Matrix NMS, to be deployed friendly on various hardware.

<div align=center>
<img src="https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/images/ppyoloe_plus_map_fps.png"/>
</div>

## Results and models

### PPYOLOE+ COCO

|  Backbone   | Arch | Size | Epoch | SyncBN | AMP | Mem (GB) | Box AP |                          Config                           |        Download        |
| :---------: | :--: | :--: | :---: | :----: | :-: | :------: | :----: | :-------------------------------------------------------: | :--------------------: |
| PPYOLOE+ -s |  P5  | 640  |  80   |  Yes   | Yes |   xxx    |  xxxx  | [config](../ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco.py) | [model](x) \| [log](x) |
| PPYOLOE+ -m |  P5  | 640  |  80   |  Yes   | Yes |   xxx    |  xxxx  | [config](../ppyoloe/ppyoloe_plus_m_fast_8xb8-80e_coco.py) | [model](x) \| [log](x) |
| PPYOLOE+ -l |  P5  | 640  |  80   |  Yes   | Yes |   xxx    |  xxxx  | [config](../ppyoloe/ppyoloe_plus_l_fast_8xb8-80e_coco.py) | [model](x) \| [log](x) |
| PPYOLOE+ -x |  P5  | 640  |  80   |  Yes   | Yes |   xxx    |  xxxx  | [config](../ppyoloe/ppyoloe_plus_x_fast_8xb8-80e_coco.py) | [model](x) \| [log](x) |

**Note**:

### PPYOLOE COCO

|  Backbone  | Arch | Size | Epoch | SyncBN | AMP | Mem (GB) | Box AP |                         Config                         |        Download        |
| :--------: | :--: | :--: | :---: | :----: | :-: | :------: | :----: | :----------------------------------------------------: | :--------------------: |
| PPYOLOE -s |  P5  | 640  |  400  |  Yes   | Yes |   xxx    |  xxxx  | [config](../ppyoloe/ppyoloe_s_fast_8xb32-400e_coco.py) | [model](x) \| [log](x) |
| PPYOLOE -s |  P5  | 640  |  300  |  Yes   | Yes |   xxx    |  xxxx  | [config](../ppyoloe/ppyoloe_s_fast_8xb32-300e_coco.py) | [model](x) \| [log](x) |
| PPYOLOE -m |  P5  | 640  |  300  |  Yes   | Yes |   xxx    |  xxxx  | [config](../ppyoloe/ppyoloe_m_fast_8xb28-300e_coco.py) | [model](x) \| [log](x) |
| PPYOLOE -l |  P5  | 640  |  300  |  Yes   | Yes |   xxx    |  xxxx  | [config](../ppyoloe/ppyoloe_l_fast_8xb20-300e_coco.py) | [model](x) \| [log](x) |
| PPYOLOE -x |  P5  | 640  |  300  |  Yes   | Yes |   xxx    |  xxxx  | [config](../ppyoloe/ppyoloe_x_fast_8xb16-300e_coco.py) | [model](x) \| [log](x) |

```latex
@article{Xu2022PPYOLOEAE,
  title={PP-YOLOE: An evolved version of YOLO},
  author={Shangliang Xu and Xinxin Wang and Wenyu Lv and Qinyao Chang and Cheng Cui and Kaipeng Deng and Guanzhong Wang and Qingqing Dang and Shengyun Wei and Yuning Du and Baohua Lai},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.16250}
}
```
