# Distill RTM Detectors Based on MMRazor

## Description

To further improve the model accuracy while not introducing much additional
computation cost, we apply the feature-based distillation to the training phase
of these RTM detectors. In summary,our distillation strategy are threefold:

(1) Inspired by [PKD](https://arxiv.org/abs/2207.02039), we first normalize
the intermediate feature maps to have zero mean and unit variances before calculating
the distillation loss.

(2) Inspired by [CWD](https://arxiv.org/abs/2011.13256), we adopt the channel-wise
distillation paradigm, which can pay more attention to the most salient regions
of each channel.

(3) Inspired by [DAMO-YOLO](https://arxiv.org/abs/2211.15444), the distillation
process is split into two stages. 1) The teacher distills the student at the
first stage (280 epochs) on strong mosaic domain. 2) The student finetunes itself
on no masaic domain at the second stage (20 epochs).

## Configs

Here we provide detection configs and models for MMRazor in MMYOLO. For clarify,
we take `./kd_tiny_rtmdet_s_neck_300e_coco.py` as an example to show how to
distill a RTM detector based on MMRazor.

Here is the main part of `./kd_tiny_rtmdet_s_neck_300e_coco.py`.

```shell
norm_cfg = dict(type='BN', affine=False, track_running_stats=False)

distiller=dict(
    type='ConfigurableDistiller',
    student_recorders=dict(
        fpn0=dict(type='ModuleOutputs', source='neck.out_layers.0.conv'),
        fpn1=dict(type='ModuleOutputs', source='neck.out_layers.1.conv'),
        fpn2=dict(type='ModuleOutputs', source='neck.out_layers.2.conv'),
    ),
    teacher_recorders=dict(
        fpn0=dict(type='ModuleOutputs', source='neck.out_layers.0.conv'),
        fpn1=dict(type='ModuleOutputs', source='neck.out_layers.1.conv'),
        fpn2=dict(type='ModuleOutputs', source='neck.out_layers.2.conv')),
    connectors=dict(
        fpn0_s=dict(type='ConvModuleConnector', in_channel=96,
            out_channel=128, bias=False, norm_cfg=norm_cfg,
            act_cfg=None),
        fpn0_t=dict(
            type='NormConnector', in_channels=128, norm_cfg=norm_cfg),
        fpn1_s=dict(
            type='ConvModuleConnector', in_channel=96,
            out_channel=128, bias=False, norm_cfg=norm_cfg,
            act_cfg=None),
        fpn1_t=dict(
            type='NormConnector', in_channels=128, norm_cfg=norm_cfg),
        fpn2_s=dict(
            type='ConvModuleConnector', in_channel=96,
            out_channel=128, bias=False, norm_cfg=norm_cfg,
            act_cfg=None),
        fpn2_t=dict(
            type='NormConnector', in_channels=128, norm_cfg=norm_cfg)),
    distill_losses=dict(
        loss_pkd_fpn0=dict(type='ChannelWiseDivergence', loss_weight=1),
        loss_pkd_fpn1=dict(type='ChannelWiseDivergence', loss_weight=1),
        loss_pkd_fpn2=dict(type='ChannelWiseDivergence', loss_weight=1)),
    loss_forward_mappings=dict(
        loss_pkd_fpn0=dict(
            preds_S=dict(from_student=True, recorder='fpn0', connector='fpn0_s'),
            preds_T=dict(from_student=False, recorder='fpn0', connector='fpn0_t')),
        loss_pkd_fpn1=dict(
            preds_S=dict(from_student=True, recorder='fpn1', connector='fpn1_s'),
            preds_T=dict(from_student=False, recorder='fpn1', connector='fpn1_t')),
        loss_pkd_fpn2=dict(
            preds_S=dict(from_student=True, recorder='fpn2', connector='fpn2_s'),
            preds_T=dict(from_student=False, recorder='fpn2', connector='fpn2_t'))))

```

`recorders` are used to record various intermediate results during the model forward.
In this example, they can help record the output of 3 `nn.Module` of the teacher
and the student.

`connectors` are adaptive layers which usually map teacher's and students features
to the same dimension.

`distill_losses` are configs for multiple distill losses.

`loss_forward_mappings` are mappings between distill loss forward arguments and records.

In addition, the student finetunes itself on no masaic domain at the last 20 epochs,
so we add a new hook named `DistillationLossDetachHook` to stop distillation on time.
We need to add this hook to the `custom_hooks` list like this:

```shell
custom_hooks = [..., dict(type='mmrazor.DistillationLossDetachHook', detach_epoch=280)]
```

## Results and Models

| Location | Dataset |                                                      Teacher                                                      |                                                         Student                                                         | mAP  | mAP(T) | mAP(S) |                    Config                    | Download                                                                                                                                                                                             |
| :------: | :-----: | :---------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: | :--: | :----: | :----: | :------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   FPN    |  COCO   | [RTMDet-s](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py) | [RTMDet-tiny](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py) | 41.8 |  44.6  |  41.0  | [config](kd_tiny_rtmdet_s_neck_300e_coco.py) | [teacher](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-0a8c901a.pth) \|[model](<>) \| [log](<>)         |
|   FPN    |  COCO   | [RTMDet-m](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco.py) |    [RTMDet-s](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py)    | 45.7 |  49.3  |  44.6  |  [config](kd_s_rtmdet_m_neck_300e_coco.py)   | [teacher](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth)         \|[model](<>) \| [log](<>) |
|   FPN    |  COCO   | [RTMDet-l](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py) |    [RTMDet-m](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco.py)    | 50.2 |  51.4  |  49.3  |  [config](kd_m_rtmdet_l_neck_300e_coco.py)   | [teacher](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth) \|[model](<>) \| [log](<>)         |
|   FPN    |  COCO   | [RTMDet-x](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/rtmdet_x_syncbn_fast_8xb32-300e_coco.py) |    [RTMDet-l](https://github.com/open-mmlab/mmyolo/blob/main/configs/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py)    | 52.3 |  52.8  |  51.4  |  [config](kd_l_rtmdet_x_neck_300e_coco.py)   | [teacher](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_x_syncbn_fast_8xb32-300e_coco/rtmdet_x_syncbn_fast_8xb32-300e_coco_20221231_100345-b85cd476.pth) \|[model](<>) \| [log](<>)         |
