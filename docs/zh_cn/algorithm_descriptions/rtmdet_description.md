# RTMDet 原理和实现全解析

## BBox Coder

RTMDet 的 BBox Coder 采用的是 `mmdet.DistancePointBBoxCoder`。

该类的 docstring 是这样的：
> This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left, right) and decode it back to the original.
> 
> 这个编码器将 gt bboxes (x1, y1, x2, y2) 编码为 (top, bottom, left, right)，并且解码至原图像上

## Loss

loss_cls、loss_cls 
