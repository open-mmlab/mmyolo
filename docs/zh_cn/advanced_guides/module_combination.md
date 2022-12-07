# 一、应用不同的Neck和Head

1. 假设想将`YOLOv6RepPAFPN`作为`YOLOv5`的`Neck`，则配置文件如下：

   ```markdown
   _base_ = './yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
   
   model = dict(
       neck = dict(
           type = 'YOLOv6RepPAFPN',
           in_channels = [256, 512, 1024], 
           out_channels = [128, 256, 512], # 注意YOLOv6RepPAFPN的输出通道是[128, 256, 512]
           num_csp_blocks = 12,
           act_cfg = dict(type='ReLU', inplace = True),
       ),
       bbox_head = dict(
           head_module = dict(
               in_channels = [128, 256, 512])) # head部分输入通道要做相应更改
   )
   ```

2.  假设想将`YOLOv7PAFPN`作为`YOLOv5`的`Neck`，则配置文件如下：

   ```markdown
   _base_ = './yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
   
   deepen_factor = _base_.deepen_factor
   widen_factor = 0.5
   
   model = dict(
       neck = dict(
           _delete_=True, # 将 _base_ 中关于 neck 的字段删除
           type = 'YOLOv7PAFPN',
           deepen_factor = deepen_factor,
           widen_factor = widen_factor,
           upsample_feats_cat_first = False,
           in_channels = [256, 512, 1024],
           out_channels = [128, 256, 512],
           norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
           act_cfg = dict(type='SiLU', inplace=True),
       ),
       bbox_head = dict(
           head_module = dict(
               in_channels = [256, 512, 1024])) # 注意使用YOLOv7PAFPN后head部分输入通道数是neck输出通道数的两倍
   )
   ```

3.  假设想将`YOLOv6Head`作为`YOLOv5`的`Head`，则配置文件如下：

   ```markdown
   _base_ = './yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
   
   
   ```

   