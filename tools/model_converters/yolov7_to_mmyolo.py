# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import torch

convert_dict_tiny = {
    # stem
    'model.0': 'backbone.stem.0',
    'model.1': 'backbone.stem.1',

    # stage1 TinyDownSampleBlock
    'model.2': 'backbone.stage1.0.short_conv',
    'model.3': 'backbone.stage1.0.main_convs.0',
    'model.4': 'backbone.stage1.0.main_convs.1',
    'model.5': 'backbone.stage1.0.main_convs.2',
    'model.7': 'backbone.stage1.0.final_conv',

    # stage2  TinyDownSampleBlock
    'model.9': 'backbone.stage2.1.short_conv',
    'model.10': 'backbone.stage2.1.main_convs.0',
    'model.11': 'backbone.stage2.1.main_convs.1',
    'model.12': 'backbone.stage2.1.main_convs.2',
    'model.14': 'backbone.stage2.1.final_conv',

    # stage3 TinyDownSampleBlock
    'model.16': 'backbone.stage3.1.short_conv',
    'model.17': 'backbone.stage3.1.main_convs.0',
    'model.18': 'backbone.stage3.1.main_convs.1',
    'model.19': 'backbone.stage3.1.main_convs.2',
    'model.21': 'backbone.stage3.1.final_conv',

    # stage4 TinyDownSampleBlock
    'model.23': 'backbone.stage4.1.short_conv',
    'model.24': 'backbone.stage4.1.main_convs.0',
    'model.25': 'backbone.stage4.1.main_convs.1',
    'model.26': 'backbone.stage4.1.main_convs.2',
    'model.28': 'backbone.stage4.1.final_conv',

    # neck SPPCSPBlock
    'model.29': 'neck.reduce_layers.2.short_layer',
    'model.30': 'neck.reduce_layers.2.main_layers',
    'model.35': 'neck.reduce_layers.2.fuse_layers',
    'model.37': 'neck.reduce_layers.2.final_conv',
    'model.38': 'neck.upsample_layers.0.0',
    'model.40': 'neck.reduce_layers.1',
    'model.42': 'neck.top_down_layers.0.short_conv',
    'model.43': 'neck.top_down_layers.0.main_convs.0',
    'model.44': 'neck.top_down_layers.0.main_convs.1',
    'model.45': 'neck.top_down_layers.0.main_convs.2',
    'model.47': 'neck.top_down_layers.0.final_conv',
    'model.48': 'neck.upsample_layers.1.0',
    'model.50': 'neck.reduce_layers.0',
    'model.52': 'neck.top_down_layers.1.short_conv',
    'model.53': 'neck.top_down_layers.1.main_convs.0',
    'model.54': 'neck.top_down_layers.1.main_convs.1',
    'model.55': 'neck.top_down_layers.1.main_convs.2',
    'model.57': 'neck.top_down_layers.1.final_conv',
    'model.58': 'neck.downsample_layers.0',
    'model.60': 'neck.bottom_up_layers.0.short_conv',
    'model.61': 'neck.bottom_up_layers.0.main_convs.0',
    'model.62': 'neck.bottom_up_layers.0.main_convs.1',
    'model.63': 'neck.bottom_up_layers.0.main_convs.2',
    'model.65': 'neck.bottom_up_layers.0.final_conv',
    'model.66': 'neck.downsample_layers.1',
    'model.68': 'neck.bottom_up_layers.1.short_conv',
    'model.69': 'neck.bottom_up_layers.1.main_convs.0',
    'model.70': 'neck.bottom_up_layers.1.main_convs.1',
    'model.71': 'neck.bottom_up_layers.1.main_convs.2',
    'model.73': 'neck.bottom_up_layers.1.final_conv',
    'model.74': 'neck.out_layers.0',
    'model.75': 'neck.out_layers.1',
    'model.76': 'neck.out_layers.2',

    # head
    'model.77.m.0': 'bbox_head.head_module.convs_pred.0.1',
    'model.77.m.1': 'bbox_head.head_module.convs_pred.1.1',
    'model.77.m.2': 'bbox_head.head_module.convs_pred.2.1'
}

convert_dict_l = {
    # stem
    'model.0': 'backbone.stem.0',
    'model.1': 'backbone.stem.1',
    'model.2': 'backbone.stem.2',

    # stage1
    # ConvModule
    'model.3': 'backbone.stage1.0',
    # ELANBlock expand_channel_2x
    'model.4': 'backbone.stage1.1.short_conv',
    'model.5': 'backbone.stage1.1.main_conv',
    'model.6': 'backbone.stage1.1.blocks.0.0',
    'model.7': 'backbone.stage1.1.blocks.0.1',
    'model.8': 'backbone.stage1.1.blocks.1.0',
    'model.9': 'backbone.stage1.1.blocks.1.1',
    'model.11': 'backbone.stage1.1.final_conv',

    # stage2
    # MaxPoolBlock reduce_channel_2x
    'model.13': 'backbone.stage2.0.maxpool_branches.1',
    'model.14': 'backbone.stage2.0.stride_conv_branches.0',
    'model.15': 'backbone.stage2.0.stride_conv_branches.1',
    # ELANBlock expand_channel_2x
    'model.17': 'backbone.stage2.1.short_conv',
    'model.18': 'backbone.stage2.1.main_conv',
    'model.19': 'backbone.stage2.1.blocks.0.0',
    'model.20': 'backbone.stage2.1.blocks.0.1',
    'model.21': 'backbone.stage2.1.blocks.1.0',
    'model.22': 'backbone.stage2.1.blocks.1.1',
    'model.24': 'backbone.stage2.1.final_conv',

    # stage3
    # MaxPoolBlock reduce_channel_2x
    'model.26': 'backbone.stage3.0.maxpool_branches.1',
    'model.27': 'backbone.stage3.0.stride_conv_branches.0',
    'model.28': 'backbone.stage3.0.stride_conv_branches.1',
    # ELANBlock expand_channel_2x
    'model.30': 'backbone.stage3.1.short_conv',
    'model.31': 'backbone.stage3.1.main_conv',
    'model.32': 'backbone.stage3.1.blocks.0.0',
    'model.33': 'backbone.stage3.1.blocks.0.1',
    'model.34': 'backbone.stage3.1.blocks.1.0',
    'model.35': 'backbone.stage3.1.blocks.1.1',
    'model.37': 'backbone.stage3.1.final_conv',

    # stage4
    # MaxPoolBlock reduce_channel_2x
    'model.39': 'backbone.stage4.0.maxpool_branches.1',
    'model.40': 'backbone.stage4.0.stride_conv_branches.0',
    'model.41': 'backbone.stage4.0.stride_conv_branches.1',
    # ELANBlock no_change_channel
    'model.43': 'backbone.stage4.1.short_conv',
    'model.44': 'backbone.stage4.1.main_conv',
    'model.45': 'backbone.stage4.1.blocks.0.0',
    'model.46': 'backbone.stage4.1.blocks.0.1',
    'model.47': 'backbone.stage4.1.blocks.1.0',
    'model.48': 'backbone.stage4.1.blocks.1.1',
    'model.50': 'backbone.stage4.1.final_conv',

    # neck SPPCSPBlock
    'model.51.cv1': 'neck.reduce_layers.2.main_layers.0',
    'model.51.cv3': 'neck.reduce_layers.2.main_layers.1',
    'model.51.cv4': 'neck.reduce_layers.2.main_layers.2',
    'model.51.cv5': 'neck.reduce_layers.2.fuse_layers.0',
    'model.51.cv6': 'neck.reduce_layers.2.fuse_layers.1',
    'model.51.cv2': 'neck.reduce_layers.2.short_layer',
    'model.51.cv7': 'neck.reduce_layers.2.final_conv',

    # neck
    'model.52': 'neck.upsample_layers.0.0',
    'model.54': 'neck.reduce_layers.1',

    # neck ELANBlock reduce_channel_2x
    'model.56': 'neck.top_down_layers.0.short_conv',
    'model.57': 'neck.top_down_layers.0.main_conv',
    'model.58': 'neck.top_down_layers.0.blocks.0',
    'model.59': 'neck.top_down_layers.0.blocks.1',
    'model.60': 'neck.top_down_layers.0.blocks.2',
    'model.61': 'neck.top_down_layers.0.blocks.3',
    'model.63': 'neck.top_down_layers.0.final_conv',
    'model.64': 'neck.upsample_layers.1.0',
    'model.66': 'neck.reduce_layers.0',

    # neck ELANBlock reduce_channel_2x
    'model.68': 'neck.top_down_layers.1.short_conv',
    'model.69': 'neck.top_down_layers.1.main_conv',
    'model.70': 'neck.top_down_layers.1.blocks.0',
    'model.71': 'neck.top_down_layers.1.blocks.1',
    'model.72': 'neck.top_down_layers.1.blocks.2',
    'model.73': 'neck.top_down_layers.1.blocks.3',
    'model.75': 'neck.top_down_layers.1.final_conv',

    # neck MaxPoolBlock no_change_channel
    'model.77': 'neck.downsample_layers.0.maxpool_branches.1',
    'model.78': 'neck.downsample_layers.0.stride_conv_branches.0',
    'model.79': 'neck.downsample_layers.0.stride_conv_branches.1',

    # neck ELANBlock reduce_channel_2x
    'model.81': 'neck.bottom_up_layers.0.short_conv',
    'model.82': 'neck.bottom_up_layers.0.main_conv',
    'model.83': 'neck.bottom_up_layers.0.blocks.0',
    'model.84': 'neck.bottom_up_layers.0.blocks.1',
    'model.85': 'neck.bottom_up_layers.0.blocks.2',
    'model.86': 'neck.bottom_up_layers.0.blocks.3',
    'model.88': 'neck.bottom_up_layers.0.final_conv',

    # neck MaxPoolBlock no_change_channel
    'model.90': 'neck.downsample_layers.1.maxpool_branches.1',
    'model.91': 'neck.downsample_layers.1.stride_conv_branches.0',
    'model.92': 'neck.downsample_layers.1.stride_conv_branches.1',

    # neck ELANBlock reduce_channel_2x
    'model.94': 'neck.bottom_up_layers.1.short_conv',
    'model.95': 'neck.bottom_up_layers.1.main_conv',
    'model.96': 'neck.bottom_up_layers.1.blocks.0',
    'model.97': 'neck.bottom_up_layers.1.blocks.1',
    'model.98': 'neck.bottom_up_layers.1.blocks.2',
    'model.99': 'neck.bottom_up_layers.1.blocks.3',
    'model.101': 'neck.bottom_up_layers.1.final_conv',

    # RepVGGBlock
    'model.102.rbr_dense.0': 'neck.out_layers.0.rbr_dense.conv',
    'model.102.rbr_dense.1': 'neck.out_layers.0.rbr_dense.bn',
    'model.102.rbr_1x1.0': 'neck.out_layers.0.rbr_1x1.conv',
    'model.102.rbr_1x1.1': 'neck.out_layers.0.rbr_1x1.bn',
    'model.103.rbr_dense.0': 'neck.out_layers.1.rbr_dense.conv',
    'model.103.rbr_dense.1': 'neck.out_layers.1.rbr_dense.bn',
    'model.103.rbr_1x1.0': 'neck.out_layers.1.rbr_1x1.conv',
    'model.103.rbr_1x1.1': 'neck.out_layers.1.rbr_1x1.bn',
    'model.104.rbr_dense.0': 'neck.out_layers.2.rbr_dense.conv',
    'model.104.rbr_dense.1': 'neck.out_layers.2.rbr_dense.bn',
    'model.104.rbr_1x1.0': 'neck.out_layers.2.rbr_1x1.conv',
    'model.104.rbr_1x1.1': 'neck.out_layers.2.rbr_1x1.bn',

    # head
    'model.105.m.0': 'bbox_head.head_module.convs_pred.0.1',
    'model.105.m.1': 'bbox_head.head_module.convs_pred.1.1',
    'model.105.m.2': 'bbox_head.head_module.convs_pred.2.1'
}

convert_dict_x = {
    # stem
    'model.0': 'backbone.stem.0',
    'model.1': 'backbone.stem.1',
    'model.2': 'backbone.stem.2',

    # stage1
    # ConvModule
    'model.3': 'backbone.stage1.0',
    # ELANBlock expand_channel_2x
    'model.4': 'backbone.stage1.1.short_conv',
    'model.5': 'backbone.stage1.1.main_conv',
    'model.6': 'backbone.stage1.1.blocks.0.0',
    'model.7': 'backbone.stage1.1.blocks.0.1',
    'model.8': 'backbone.stage1.1.blocks.1.0',
    'model.9': 'backbone.stage1.1.blocks.1.1',
    'model.10': 'backbone.stage1.1.blocks.2.0',
    'model.11': 'backbone.stage1.1.blocks.2.1',
    'model.13': 'backbone.stage1.1.final_conv',

    # stage2
    # MaxPoolBlock reduce_channel_2x
    'model.15': 'backbone.stage2.0.maxpool_branches.1',
    'model.16': 'backbone.stage2.0.stride_conv_branches.0',
    'model.17': 'backbone.stage2.0.stride_conv_branches.1',

    # ELANBlock expand_channel_2x
    'model.19': 'backbone.stage2.1.short_conv',
    'model.20': 'backbone.stage2.1.main_conv',
    'model.21': 'backbone.stage2.1.blocks.0.0',
    'model.22': 'backbone.stage2.1.blocks.0.1',
    'model.23': 'backbone.stage2.1.blocks.1.0',
    'model.24': 'backbone.stage2.1.blocks.1.1',
    'model.25': 'backbone.stage2.1.blocks.2.0',
    'model.26': 'backbone.stage2.1.blocks.2.1',
    'model.28': 'backbone.stage2.1.final_conv',

    # stage3
    # MaxPoolBlock reduce_channel_2x
    'model.30': 'backbone.stage3.0.maxpool_branches.1',
    'model.31': 'backbone.stage3.0.stride_conv_branches.0',
    'model.32': 'backbone.stage3.0.stride_conv_branches.1',
    # ELANBlock expand_channel_2x
    'model.34': 'backbone.stage3.1.short_conv',
    'model.35': 'backbone.stage3.1.main_conv',
    'model.36': 'backbone.stage3.1.blocks.0.0',
    'model.37': 'backbone.stage3.1.blocks.0.1',
    'model.38': 'backbone.stage3.1.blocks.1.0',
    'model.39': 'backbone.stage3.1.blocks.1.1',
    'model.40': 'backbone.stage3.1.blocks.2.0',
    'model.41': 'backbone.stage3.1.blocks.2.1',
    'model.43': 'backbone.stage3.1.final_conv',

    # stage4
    # MaxPoolBlock reduce_channel_2x
    'model.45': 'backbone.stage4.0.maxpool_branches.1',
    'model.46': 'backbone.stage4.0.stride_conv_branches.0',
    'model.47': 'backbone.stage4.0.stride_conv_branches.1',
    # ELANBlock no_change_channel
    'model.49': 'backbone.stage4.1.short_conv',
    'model.50': 'backbone.stage4.1.main_conv',
    'model.51': 'backbone.stage4.1.blocks.0.0',
    'model.52': 'backbone.stage4.1.blocks.0.1',
    'model.53': 'backbone.stage4.1.blocks.1.0',
    'model.54': 'backbone.stage4.1.blocks.1.1',
    'model.55': 'backbone.stage4.1.blocks.2.0',
    'model.56': 'backbone.stage4.1.blocks.2.1',
    'model.58': 'backbone.stage4.1.final_conv',

    # neck SPPCSPBlock
    'model.59.cv1': 'neck.reduce_layers.2.main_layers.0',
    'model.59.cv3': 'neck.reduce_layers.2.main_layers.1',
    'model.59.cv4': 'neck.reduce_layers.2.main_layers.2',
    'model.59.cv5': 'neck.reduce_layers.2.fuse_layers.0',
    'model.59.cv6': 'neck.reduce_layers.2.fuse_layers.1',
    'model.59.cv2': 'neck.reduce_layers.2.short_layer',
    'model.59.cv7': 'neck.reduce_layers.2.final_conv',

    # neck
    'model.60': 'neck.upsample_layers.0.0',
    'model.62': 'neck.reduce_layers.1',

    # neck ELANBlock reduce_channel_2x
    'model.64': 'neck.top_down_layers.0.short_conv',
    'model.65': 'neck.top_down_layers.0.main_conv',
    'model.66': 'neck.top_down_layers.0.blocks.0.0',
    'model.67': 'neck.top_down_layers.0.blocks.0.1',
    'model.68': 'neck.top_down_layers.0.blocks.1.0',
    'model.69': 'neck.top_down_layers.0.blocks.1.1',
    'model.70': 'neck.top_down_layers.0.blocks.2.0',
    'model.71': 'neck.top_down_layers.0.blocks.2.1',
    'model.73': 'neck.top_down_layers.0.final_conv',
    'model.74': 'neck.upsample_layers.1.0',
    'model.76': 'neck.reduce_layers.0',

    # neck ELANBlock reduce_channel_2x
    'model.78': 'neck.top_down_layers.1.short_conv',
    'model.79': 'neck.top_down_layers.1.main_conv',
    'model.80': 'neck.top_down_layers.1.blocks.0.0',
    'model.81': 'neck.top_down_layers.1.blocks.0.1',
    'model.82': 'neck.top_down_layers.1.blocks.1.0',
    'model.83': 'neck.top_down_layers.1.blocks.1.1',
    'model.84': 'neck.top_down_layers.1.blocks.2.0',
    'model.85': 'neck.top_down_layers.1.blocks.2.1',
    'model.87': 'neck.top_down_layers.1.final_conv',

    # neck MaxPoolBlock no_change_channel
    'model.89': 'neck.downsample_layers.0.maxpool_branches.1',
    'model.90': 'neck.downsample_layers.0.stride_conv_branches.0',
    'model.91': 'neck.downsample_layers.0.stride_conv_branches.1',

    # neck ELANBlock reduce_channel_2x
    'model.93': 'neck.bottom_up_layers.0.short_conv',
    'model.94': 'neck.bottom_up_layers.0.main_conv',
    'model.95': 'neck.bottom_up_layers.0.blocks.0.0',
    'model.96': 'neck.bottom_up_layers.0.blocks.0.1',
    'model.97': 'neck.bottom_up_layers.0.blocks.1.0',
    'model.98': 'neck.bottom_up_layers.0.blocks.1.1',
    'model.99': 'neck.bottom_up_layers.0.blocks.2.0',
    'model.100': 'neck.bottom_up_layers.0.blocks.2.1',
    'model.102': 'neck.bottom_up_layers.0.final_conv',

    # neck MaxPoolBlock no_change_channel
    'model.104': 'neck.downsample_layers.1.maxpool_branches.1',
    'model.105': 'neck.downsample_layers.1.stride_conv_branches.0',
    'model.106': 'neck.downsample_layers.1.stride_conv_branches.1',

    # neck ELANBlock reduce_channel_2x
    'model.108': 'neck.bottom_up_layers.1.short_conv',
    'model.109': 'neck.bottom_up_layers.1.main_conv',
    'model.110': 'neck.bottom_up_layers.1.blocks.0.0',
    'model.111': 'neck.bottom_up_layers.1.blocks.0.1',
    'model.112': 'neck.bottom_up_layers.1.blocks.1.0',
    'model.113': 'neck.bottom_up_layers.1.blocks.1.1',
    'model.114': 'neck.bottom_up_layers.1.blocks.2.0',
    'model.115': 'neck.bottom_up_layers.1.blocks.2.1',
    'model.117': 'neck.bottom_up_layers.1.final_conv',

    # Conv
    'model.118': 'neck.out_layers.0',
    'model.119': 'neck.out_layers.1',
    'model.120': 'neck.out_layers.2',

    # head
    'model.121.m.0': 'bbox_head.head_module.convs_pred.0.1',
    'model.121.m.1': 'bbox_head.head_module.convs_pred.1.1',
    'model.121.m.2': 'bbox_head.head_module.convs_pred.2.1'
}

convert_dict_w = {
    # stem
    'model.1': 'backbone.stem.conv',

    # stage1
    # ConvModule
    'model.2': 'backbone.stage1.0',
    # ELANBlock
    'model.3': 'backbone.stage1.1.short_conv',
    'model.4': 'backbone.stage1.1.main_conv',
    'model.5': 'backbone.stage1.1.blocks.0.0',
    'model.6': 'backbone.stage1.1.blocks.0.1',
    'model.7': 'backbone.stage1.1.blocks.1.0',
    'model.8': 'backbone.stage1.1.blocks.1.1',
    'model.10': 'backbone.stage1.1.final_conv',

    # stage2
    'model.11': 'backbone.stage2.0',
    # ELANBlock
    'model.12': 'backbone.stage2.1.short_conv',
    'model.13': 'backbone.stage2.1.main_conv',
    'model.14': 'backbone.stage2.1.blocks.0.0',
    'model.15': 'backbone.stage2.1.blocks.0.1',
    'model.16': 'backbone.stage2.1.blocks.1.0',
    'model.17': 'backbone.stage2.1.blocks.1.1',
    'model.19': 'backbone.stage2.1.final_conv',

    # stage3
    'model.20': 'backbone.stage3.0',
    # ELANBlock
    'model.21': 'backbone.stage3.1.short_conv',
    'model.22': 'backbone.stage3.1.main_conv',
    'model.23': 'backbone.stage3.1.blocks.0.0',
    'model.24': 'backbone.stage3.1.blocks.0.1',
    'model.25': 'backbone.stage3.1.blocks.1.0',
    'model.26': 'backbone.stage3.1.blocks.1.1',
    'model.28': 'backbone.stage3.1.final_conv',

    # stage4
    'model.29': 'backbone.stage4.0',
    # ELANBlock
    'model.30': 'backbone.stage4.1.short_conv',
    'model.31': 'backbone.stage4.1.main_conv',
    'model.32': 'backbone.stage4.1.blocks.0.0',
    'model.33': 'backbone.stage4.1.blocks.0.1',
    'model.34': 'backbone.stage4.1.blocks.1.0',
    'model.35': 'backbone.stage4.1.blocks.1.1',
    'model.37': 'backbone.stage4.1.final_conv',

    # stage5
    'model.38': 'backbone.stage5.0',
    # ELANBlock
    'model.39': 'backbone.stage5.1.short_conv',
    'model.40': 'backbone.stage5.1.main_conv',
    'model.41': 'backbone.stage5.1.blocks.0.0',
    'model.42': 'backbone.stage5.1.blocks.0.1',
    'model.43': 'backbone.stage5.1.blocks.1.0',
    'model.44': 'backbone.stage5.1.blocks.1.1',
    'model.46': 'backbone.stage5.1.final_conv',

    # neck SPPCSPBlock
    'model.47.cv1': 'neck.reduce_layers.3.main_layers.0',
    'model.47.cv3': 'neck.reduce_layers.3.main_layers.1',
    'model.47.cv4': 'neck.reduce_layers.3.main_layers.2',
    'model.47.cv5': 'neck.reduce_layers.3.fuse_layers.0',
    'model.47.cv6': 'neck.reduce_layers.3.fuse_layers.1',
    'model.47.cv2': 'neck.reduce_layers.3.short_layer',
    'model.47.cv7': 'neck.reduce_layers.3.final_conv',

    # neck
    'model.48': 'neck.upsample_layers.0.0',
    'model.50': 'neck.reduce_layers.2',

    # neck ELANBlock
    'model.52': 'neck.top_down_layers.0.short_conv',
    'model.53': 'neck.top_down_layers.0.main_conv',
    'model.54': 'neck.top_down_layers.0.blocks.0',
    'model.55': 'neck.top_down_layers.0.blocks.1',
    'model.56': 'neck.top_down_layers.0.blocks.2',
    'model.57': 'neck.top_down_layers.0.blocks.3',
    'model.59': 'neck.top_down_layers.0.final_conv',
    'model.60': 'neck.upsample_layers.1.0',
    'model.62': 'neck.reduce_layers.1',

    # neck ELANBlock reduce_channel_2x
    'model.64': 'neck.top_down_layers.1.short_conv',
    'model.65': 'neck.top_down_layers.1.main_conv',
    'model.66': 'neck.top_down_layers.1.blocks.0',
    'model.67': 'neck.top_down_layers.1.blocks.1',
    'model.68': 'neck.top_down_layers.1.blocks.2',
    'model.69': 'neck.top_down_layers.1.blocks.3',
    'model.71': 'neck.top_down_layers.1.final_conv',
    'model.72': 'neck.upsample_layers.2.0',
    'model.74': 'neck.reduce_layers.0',
    'model.76': 'neck.top_down_layers.2.short_conv',
    'model.77': 'neck.top_down_layers.2.main_conv',
    'model.78': 'neck.top_down_layers.2.blocks.0',
    'model.79': 'neck.top_down_layers.2.blocks.1',
    'model.80': 'neck.top_down_layers.2.blocks.2',
    'model.81': 'neck.top_down_layers.2.blocks.3',
    'model.83': 'neck.top_down_layers.2.final_conv',
    'model.84': 'neck.downsample_layers.0',

    # neck ELANBlock
    'model.86': 'neck.bottom_up_layers.0.short_conv',
    'model.87': 'neck.bottom_up_layers.0.main_conv',
    'model.88': 'neck.bottom_up_layers.0.blocks.0',
    'model.89': 'neck.bottom_up_layers.0.blocks.1',
    'model.90': 'neck.bottom_up_layers.0.blocks.2',
    'model.91': 'neck.bottom_up_layers.0.blocks.3',
    'model.93': 'neck.bottom_up_layers.0.final_conv',
    'model.94': 'neck.downsample_layers.1',

    # neck ELANBlock reduce_channel_2x
    'model.96': 'neck.bottom_up_layers.1.short_conv',
    'model.97': 'neck.bottom_up_layers.1.main_conv',
    'model.98': 'neck.bottom_up_layers.1.blocks.0',
    'model.99': 'neck.bottom_up_layers.1.blocks.1',
    'model.100': 'neck.bottom_up_layers.1.blocks.2',
    'model.101': 'neck.bottom_up_layers.1.blocks.3',
    'model.103': 'neck.bottom_up_layers.1.final_conv',
    'model.104': 'neck.downsample_layers.2',

    # neck ELANBlock reduce_channel_2x
    'model.106': 'neck.bottom_up_layers.2.short_conv',
    'model.107': 'neck.bottom_up_layers.2.main_conv',
    'model.108': 'neck.bottom_up_layers.2.blocks.0',
    'model.109': 'neck.bottom_up_layers.2.blocks.1',
    'model.110': 'neck.bottom_up_layers.2.blocks.2',
    'model.111': 'neck.bottom_up_layers.2.blocks.3',
    'model.113': 'neck.bottom_up_layers.2.final_conv',
    'model.114': 'bbox_head.head_module.main_convs_pred.0.0',
    'model.115': 'bbox_head.head_module.main_convs_pred.1.0',
    'model.116': 'bbox_head.head_module.main_convs_pred.2.0',
    'model.117': 'bbox_head.head_module.main_convs_pred.3.0',

    # head
    'model.118.m.0': 'bbox_head.head_module.main_convs_pred.0.2',
    'model.118.m.1': 'bbox_head.head_module.main_convs_pred.1.2',
    'model.118.m.2': 'bbox_head.head_module.main_convs_pred.2.2',
    'model.118.m.3': 'bbox_head.head_module.main_convs_pred.3.2'
}

convert_dict_e = {
    # stem
    'model.1': 'backbone.stem.conv',

    # stage1
    'model.2.cv1': 'backbone.stage1.0.stride_conv_branches.0',
    'model.2.cv2': 'backbone.stage1.0.stride_conv_branches.1',
    'model.2.cv3': 'backbone.stage1.0.maxpool_branches.1',

    # ELANBlock
    'model.3': 'backbone.stage1.1.short_conv',
    'model.4': 'backbone.stage1.1.main_conv',
    'model.5': 'backbone.stage1.1.blocks.0.0',
    'model.6': 'backbone.stage1.1.blocks.0.1',
    'model.7': 'backbone.stage1.1.blocks.1.0',
    'model.8': 'backbone.stage1.1.blocks.1.1',
    'model.9': 'backbone.stage1.1.blocks.2.0',
    'model.10': 'backbone.stage1.1.blocks.2.1',
    'model.12': 'backbone.stage1.1.final_conv',

    # stage2
    'model.13.cv1': 'backbone.stage2.0.stride_conv_branches.0',
    'model.13.cv2': 'backbone.stage2.0.stride_conv_branches.1',
    'model.13.cv3': 'backbone.stage2.0.maxpool_branches.1',

    # ELANBlock
    'model.14': 'backbone.stage2.1.short_conv',
    'model.15': 'backbone.stage2.1.main_conv',
    'model.16': 'backbone.stage2.1.blocks.0.0',
    'model.17': 'backbone.stage2.1.blocks.0.1',
    'model.18': 'backbone.stage2.1.blocks.1.0',
    'model.19': 'backbone.stage2.1.blocks.1.1',
    'model.20': 'backbone.stage2.1.blocks.2.0',
    'model.21': 'backbone.stage2.1.blocks.2.1',
    'model.23': 'backbone.stage2.1.final_conv',

    # stage3
    'model.24.cv1': 'backbone.stage3.0.stride_conv_branches.0',
    'model.24.cv2': 'backbone.stage3.0.stride_conv_branches.1',
    'model.24.cv3': 'backbone.stage3.0.maxpool_branches.1',

    # ELANBlock
    'model.25': 'backbone.stage3.1.short_conv',
    'model.26': 'backbone.stage3.1.main_conv',
    'model.27': 'backbone.stage3.1.blocks.0.0',
    'model.28': 'backbone.stage3.1.blocks.0.1',
    'model.29': 'backbone.stage3.1.blocks.1.0',
    'model.30': 'backbone.stage3.1.blocks.1.1',
    'model.31': 'backbone.stage3.1.blocks.2.0',
    'model.32': 'backbone.stage3.1.blocks.2.1',
    'model.34': 'backbone.stage3.1.final_conv',

    # stage4
    'model.35.cv1': 'backbone.stage4.0.stride_conv_branches.0',
    'model.35.cv2': 'backbone.stage4.0.stride_conv_branches.1',
    'model.35.cv3': 'backbone.stage4.0.maxpool_branches.1',

    # ELANBlock
    'model.36': 'backbone.stage4.1.short_conv',
    'model.37': 'backbone.stage4.1.main_conv',
    'model.38': 'backbone.stage4.1.blocks.0.0',
    'model.39': 'backbone.stage4.1.blocks.0.1',
    'model.40': 'backbone.stage4.1.blocks.1.0',
    'model.41': 'backbone.stage4.1.blocks.1.1',
    'model.42': 'backbone.stage4.1.blocks.2.0',
    'model.43': 'backbone.stage4.1.blocks.2.1',
    'model.45': 'backbone.stage4.1.final_conv',

    # stage5
    'model.46.cv1': 'backbone.stage5.0.stride_conv_branches.0',
    'model.46.cv2': 'backbone.stage5.0.stride_conv_branches.1',
    'model.46.cv3': 'backbone.stage5.0.maxpool_branches.1',

    # ELANBlock
    'model.47': 'backbone.stage5.1.short_conv',
    'model.48': 'backbone.stage5.1.main_conv',
    'model.49': 'backbone.stage5.1.blocks.0.0',
    'model.50': 'backbone.stage5.1.blocks.0.1',
    'model.51': 'backbone.stage5.1.blocks.1.0',
    'model.52': 'backbone.stage5.1.blocks.1.1',
    'model.53': 'backbone.stage5.1.blocks.2.0',
    'model.54': 'backbone.stage5.1.blocks.2.1',
    'model.56': 'backbone.stage5.1.final_conv',

    # neck SPPCSPBlock
    'model.57.cv1': 'neck.reduce_layers.3.main_layers.0',
    'model.57.cv3': 'neck.reduce_layers.3.main_layers.1',
    'model.57.cv4': 'neck.reduce_layers.3.main_layers.2',
    'model.57.cv5': 'neck.reduce_layers.3.fuse_layers.0',
    'model.57.cv6': 'neck.reduce_layers.3.fuse_layers.1',
    'model.57.cv2': 'neck.reduce_layers.3.short_layer',
    'model.57.cv7': 'neck.reduce_layers.3.final_conv',

    # neck
    'model.58': 'neck.upsample_layers.0.0',
    'model.60': 'neck.reduce_layers.2',

    # neck ELANBlock
    'model.62': 'neck.top_down_layers.0.short_conv',
    'model.63': 'neck.top_down_layers.0.main_conv',
    'model.64': 'neck.top_down_layers.0.blocks.0',
    'model.65': 'neck.top_down_layers.0.blocks.1',
    'model.66': 'neck.top_down_layers.0.blocks.2',
    'model.67': 'neck.top_down_layers.0.blocks.3',
    'model.68': 'neck.top_down_layers.0.blocks.4',
    'model.69': 'neck.top_down_layers.0.blocks.5',
    'model.71': 'neck.top_down_layers.0.final_conv',
    'model.72': 'neck.upsample_layers.1.0',
    'model.74': 'neck.reduce_layers.1',

    # neck ELANBlock
    'model.76': 'neck.top_down_layers.1.short_conv',
    'model.77': 'neck.top_down_layers.1.main_conv',
    'model.78': 'neck.top_down_layers.1.blocks.0',
    'model.79': 'neck.top_down_layers.1.blocks.1',
    'model.80': 'neck.top_down_layers.1.blocks.2',
    'model.81': 'neck.top_down_layers.1.blocks.3',
    'model.82': 'neck.top_down_layers.1.blocks.4',
    'model.83': 'neck.top_down_layers.1.blocks.5',
    'model.85': 'neck.top_down_layers.1.final_conv',
    'model.86': 'neck.upsample_layers.2.0',
    'model.88': 'neck.reduce_layers.0',
    'model.90': 'neck.top_down_layers.2.short_conv',
    'model.91': 'neck.top_down_layers.2.main_conv',
    'model.92': 'neck.top_down_layers.2.blocks.0',
    'model.93': 'neck.top_down_layers.2.blocks.1',
    'model.94': 'neck.top_down_layers.2.blocks.2',
    'model.95': 'neck.top_down_layers.2.blocks.3',
    'model.96': 'neck.top_down_layers.2.blocks.4',
    'model.97': 'neck.top_down_layers.2.blocks.5',
    'model.99': 'neck.top_down_layers.2.final_conv',
    'model.100.cv1': 'neck.downsample_layers.0.stride_conv_branches.0',
    'model.100.cv2': 'neck.downsample_layers.0.stride_conv_branches.1',
    'model.100.cv3': 'neck.downsample_layers.0.maxpool_branches.1',

    # neck ELANBlock
    'model.102': 'neck.bottom_up_layers.0.short_conv',
    'model.103': 'neck.bottom_up_layers.0.main_conv',
    'model.104': 'neck.bottom_up_layers.0.blocks.0',
    'model.105': 'neck.bottom_up_layers.0.blocks.1',
    'model.106': 'neck.bottom_up_layers.0.blocks.2',
    'model.107': 'neck.bottom_up_layers.0.blocks.3',
    'model.108': 'neck.bottom_up_layers.0.blocks.4',
    'model.109': 'neck.bottom_up_layers.0.blocks.5',
    'model.111': 'neck.bottom_up_layers.0.final_conv',
    'model.112.cv1': 'neck.downsample_layers.1.stride_conv_branches.0',
    'model.112.cv2': 'neck.downsample_layers.1.stride_conv_branches.1',
    'model.112.cv3': 'neck.downsample_layers.1.maxpool_branches.1',

    # neck ELANBlock
    'model.114': 'neck.bottom_up_layers.1.short_conv',
    'model.115': 'neck.bottom_up_layers.1.main_conv',
    'model.116': 'neck.bottom_up_layers.1.blocks.0',
    'model.117': 'neck.bottom_up_layers.1.blocks.1',
    'model.118': 'neck.bottom_up_layers.1.blocks.2',
    'model.119': 'neck.bottom_up_layers.1.blocks.3',
    'model.120': 'neck.bottom_up_layers.1.blocks.4',
    'model.121': 'neck.bottom_up_layers.1.blocks.5',
    'model.123': 'neck.bottom_up_layers.1.final_conv',
    'model.124.cv1': 'neck.downsample_layers.2.stride_conv_branches.0',
    'model.124.cv2': 'neck.downsample_layers.2.stride_conv_branches.1',
    'model.124.cv3': 'neck.downsample_layers.2.maxpool_branches.1',

    # neck ELANBlock
    'model.126': 'neck.bottom_up_layers.2.short_conv',
    'model.127': 'neck.bottom_up_layers.2.main_conv',
    'model.128': 'neck.bottom_up_layers.2.blocks.0',
    'model.129': 'neck.bottom_up_layers.2.blocks.1',
    'model.130': 'neck.bottom_up_layers.2.blocks.2',
    'model.131': 'neck.bottom_up_layers.2.blocks.3',
    'model.132': 'neck.bottom_up_layers.2.blocks.4',
    'model.133': 'neck.bottom_up_layers.2.blocks.5',
    'model.135': 'neck.bottom_up_layers.2.final_conv',
    'model.136': 'bbox_head.head_module.main_convs_pred.0.0',
    'model.137': 'bbox_head.head_module.main_convs_pred.1.0',
    'model.138': 'bbox_head.head_module.main_convs_pred.2.0',
    'model.139': 'bbox_head.head_module.main_convs_pred.3.0',

    # head
    'model.140.m.0': 'bbox_head.head_module.main_convs_pred.0.2',
    'model.140.m.1': 'bbox_head.head_module.main_convs_pred.1.2',
    'model.140.m.2': 'bbox_head.head_module.main_convs_pred.2.2',
    'model.140.m.3': 'bbox_head.head_module.main_convs_pred.3.2'
}

convert_dict_e2e = {
    # stem
    'model.1': 'backbone.stem.conv',

    # stage1
    'model.2.cv1': 'backbone.stage1.0.stride_conv_branches.0',
    'model.2.cv2': 'backbone.stage1.0.stride_conv_branches.1',
    'model.2.cv3': 'backbone.stage1.0.maxpool_branches.1',

    # E-ELANBlock
    'model.3': 'backbone.stage1.1.e_elan_blocks.0.short_conv',
    'model.4': 'backbone.stage1.1.e_elan_blocks.0.main_conv',
    'model.5': 'backbone.stage1.1.e_elan_blocks.0.blocks.0.0',
    'model.6': 'backbone.stage1.1.e_elan_blocks.0.blocks.0.1',
    'model.7': 'backbone.stage1.1.e_elan_blocks.0.blocks.1.0',
    'model.8': 'backbone.stage1.1.e_elan_blocks.0.blocks.1.1',
    'model.9': 'backbone.stage1.1.e_elan_blocks.0.blocks.2.0',
    'model.10': 'backbone.stage1.1.e_elan_blocks.0.blocks.2.1',
    'model.12': 'backbone.stage1.1.e_elan_blocks.0.final_conv',
    'model.13': 'backbone.stage1.1.e_elan_blocks.1.short_conv',
    'model.14': 'backbone.stage1.1.e_elan_blocks.1.main_conv',
    'model.15': 'backbone.stage1.1.e_elan_blocks.1.blocks.0.0',
    'model.16': 'backbone.stage1.1.e_elan_blocks.1.blocks.0.1',
    'model.17': 'backbone.stage1.1.e_elan_blocks.1.blocks.1.0',
    'model.18': 'backbone.stage1.1.e_elan_blocks.1.blocks.1.1',
    'model.19': 'backbone.stage1.1.e_elan_blocks.1.blocks.2.0',
    'model.20': 'backbone.stage1.1.e_elan_blocks.1.blocks.2.1',
    'model.22': 'backbone.stage1.1.e_elan_blocks.1.final_conv',

    # stage2
    'model.24.cv1': 'backbone.stage2.0.stride_conv_branches.0',
    'model.24.cv2': 'backbone.stage2.0.stride_conv_branches.1',
    'model.24.cv3': 'backbone.stage2.0.maxpool_branches.1',

    # E-ELANBlock
    'model.25': 'backbone.stage2.1.e_elan_blocks.0.short_conv',
    'model.26': 'backbone.stage2.1.e_elan_blocks.0.main_conv',
    'model.27': 'backbone.stage2.1.e_elan_blocks.0.blocks.0.0',
    'model.28': 'backbone.stage2.1.e_elan_blocks.0.blocks.0.1',
    'model.29': 'backbone.stage2.1.e_elan_blocks.0.blocks.1.0',
    'model.30': 'backbone.stage2.1.e_elan_blocks.0.blocks.1.1',
    'model.31': 'backbone.stage2.1.e_elan_blocks.0.blocks.2.0',
    'model.32': 'backbone.stage2.1.e_elan_blocks.0.blocks.2.1',
    'model.34': 'backbone.stage2.1.e_elan_blocks.0.final_conv',
    'model.35': 'backbone.stage2.1.e_elan_blocks.1.short_conv',
    'model.36': 'backbone.stage2.1.e_elan_blocks.1.main_conv',
    'model.37': 'backbone.stage2.1.e_elan_blocks.1.blocks.0.0',
    'model.38': 'backbone.stage2.1.e_elan_blocks.1.blocks.0.1',
    'model.39': 'backbone.stage2.1.e_elan_blocks.1.blocks.1.0',
    'model.40': 'backbone.stage2.1.e_elan_blocks.1.blocks.1.1',
    'model.41': 'backbone.stage2.1.e_elan_blocks.1.blocks.2.0',
    'model.42': 'backbone.stage2.1.e_elan_blocks.1.blocks.2.1',
    'model.44': 'backbone.stage2.1.e_elan_blocks.1.final_conv',

    # stage3
    'model.46.cv1': 'backbone.stage3.0.stride_conv_branches.0',
    'model.46.cv2': 'backbone.stage3.0.stride_conv_branches.1',
    'model.46.cv3': 'backbone.stage3.0.maxpool_branches.1',

    # E-ELANBlock
    'model.47': 'backbone.stage3.1.e_elan_blocks.0.short_conv',
    'model.48': 'backbone.stage3.1.e_elan_blocks.0.main_conv',
    'model.49': 'backbone.stage3.1.e_elan_blocks.0.blocks.0.0',
    'model.50': 'backbone.stage3.1.e_elan_blocks.0.blocks.0.1',
    'model.51': 'backbone.stage3.1.e_elan_blocks.0.blocks.1.0',
    'model.52': 'backbone.stage3.1.e_elan_blocks.0.blocks.1.1',
    'model.53': 'backbone.stage3.1.e_elan_blocks.0.blocks.2.0',
    'model.54': 'backbone.stage3.1.e_elan_blocks.0.blocks.2.1',
    'model.56': 'backbone.stage3.1.e_elan_blocks.0.final_conv',
    'model.57': 'backbone.stage3.1.e_elan_blocks.1.short_conv',
    'model.58': 'backbone.stage3.1.e_elan_blocks.1.main_conv',
    'model.59': 'backbone.stage3.1.e_elan_blocks.1.blocks.0.0',
    'model.60': 'backbone.stage3.1.e_elan_blocks.1.blocks.0.1',
    'model.61': 'backbone.stage3.1.e_elan_blocks.1.blocks.1.0',
    'model.62': 'backbone.stage3.1.e_elan_blocks.1.blocks.1.1',
    'model.63': 'backbone.stage3.1.e_elan_blocks.1.blocks.2.0',
    'model.64': 'backbone.stage3.1.e_elan_blocks.1.blocks.2.1',
    'model.66': 'backbone.stage3.1.e_elan_blocks.1.final_conv',

    # stage4
    'model.68.cv1': 'backbone.stage4.0.stride_conv_branches.0',
    'model.68.cv2': 'backbone.stage4.0.stride_conv_branches.1',
    'model.68.cv3': 'backbone.stage4.0.maxpool_branches.1',

    # E-ELANBlock
    'model.69': 'backbone.stage4.1.e_elan_blocks.0.short_conv',
    'model.70': 'backbone.stage4.1.e_elan_blocks.0.main_conv',
    'model.71': 'backbone.stage4.1.e_elan_blocks.0.blocks.0.0',
    'model.72': 'backbone.stage4.1.e_elan_blocks.0.blocks.0.1',
    'model.73': 'backbone.stage4.1.e_elan_blocks.0.blocks.1.0',
    'model.74': 'backbone.stage4.1.e_elan_blocks.0.blocks.1.1',
    'model.75': 'backbone.stage4.1.e_elan_blocks.0.blocks.2.0',
    'model.76': 'backbone.stage4.1.e_elan_blocks.0.blocks.2.1',
    'model.78': 'backbone.stage4.1.e_elan_blocks.0.final_conv',
    'model.79': 'backbone.stage4.1.e_elan_blocks.1.short_conv',
    'model.80': 'backbone.stage4.1.e_elan_blocks.1.main_conv',
    'model.81': 'backbone.stage4.1.e_elan_blocks.1.blocks.0.0',
    'model.82': 'backbone.stage4.1.e_elan_blocks.1.blocks.0.1',
    'model.83': 'backbone.stage4.1.e_elan_blocks.1.blocks.1.0',
    'model.84': 'backbone.stage4.1.e_elan_blocks.1.blocks.1.1',
    'model.85': 'backbone.stage4.1.e_elan_blocks.1.blocks.2.0',
    'model.86': 'backbone.stage4.1.e_elan_blocks.1.blocks.2.1',
    'model.88': 'backbone.stage4.1.e_elan_blocks.1.final_conv',

    # stage5
    'model.90.cv1': 'backbone.stage5.0.stride_conv_branches.0',
    'model.90.cv2': 'backbone.stage5.0.stride_conv_branches.1',
    'model.90.cv3': 'backbone.stage5.0.maxpool_branches.1',

    # E-ELANBlock
    'model.91': 'backbone.stage5.1.e_elan_blocks.0.short_conv',
    'model.92': 'backbone.stage5.1.e_elan_blocks.0.main_conv',
    'model.93': 'backbone.stage5.1.e_elan_blocks.0.blocks.0.0',
    'model.94': 'backbone.stage5.1.e_elan_blocks.0.blocks.0.1',
    'model.95': 'backbone.stage5.1.e_elan_blocks.0.blocks.1.0',
    'model.96': 'backbone.stage5.1.e_elan_blocks.0.blocks.1.1',
    'model.97': 'backbone.stage5.1.e_elan_blocks.0.blocks.2.0',
    'model.98': 'backbone.stage5.1.e_elan_blocks.0.blocks.2.1',
    'model.100': 'backbone.stage5.1.e_elan_blocks.0.final_conv',
    'model.101': 'backbone.stage5.1.e_elan_blocks.1.short_conv',
    'model.102': 'backbone.stage5.1.e_elan_blocks.1.main_conv',
    'model.103': 'backbone.stage5.1.e_elan_blocks.1.blocks.0.0',
    'model.104': 'backbone.stage5.1.e_elan_blocks.1.blocks.0.1',
    'model.105': 'backbone.stage5.1.e_elan_blocks.1.blocks.1.0',
    'model.106': 'backbone.stage5.1.e_elan_blocks.1.blocks.1.1',
    'model.107': 'backbone.stage5.1.e_elan_blocks.1.blocks.2.0',
    'model.108': 'backbone.stage5.1.e_elan_blocks.1.blocks.2.1',
    'model.110': 'backbone.stage5.1.e_elan_blocks.1.final_conv',

    # neck SPPCSPBlock
    'model.112.cv1': 'neck.reduce_layers.3.main_layers.0',
    'model.112.cv3': 'neck.reduce_layers.3.main_layers.1',
    'model.112.cv4': 'neck.reduce_layers.3.main_layers.2',
    'model.112.cv5': 'neck.reduce_layers.3.fuse_layers.0',
    'model.112.cv6': 'neck.reduce_layers.3.fuse_layers.1',
    'model.112.cv2': 'neck.reduce_layers.3.short_layer',
    'model.112.cv7': 'neck.reduce_layers.3.final_conv',

    # neck
    'model.113': 'neck.upsample_layers.0.0',
    'model.115': 'neck.reduce_layers.2',

    # neck E-ELANBlock
    'model.117': 'neck.top_down_layers.0.e_elan_blocks.0.short_conv',
    'model.118': 'neck.top_down_layers.0.e_elan_blocks.0.main_conv',
    'model.119': 'neck.top_down_layers.0.e_elan_blocks.0.blocks.0',
    'model.120': 'neck.top_down_layers.0.e_elan_blocks.0.blocks.1',
    'model.121': 'neck.top_down_layers.0.e_elan_blocks.0.blocks.2',
    'model.122': 'neck.top_down_layers.0.e_elan_blocks.0.blocks.3',
    'model.123': 'neck.top_down_layers.0.e_elan_blocks.0.blocks.4',
    'model.124': 'neck.top_down_layers.0.e_elan_blocks.0.blocks.5',
    'model.126': 'neck.top_down_layers.0.e_elan_blocks.0.final_conv',
    'model.127': 'neck.top_down_layers.0.e_elan_blocks.1.short_conv',
    'model.128': 'neck.top_down_layers.0.e_elan_blocks.1.main_conv',
    'model.129': 'neck.top_down_layers.0.e_elan_blocks.1.blocks.0',
    'model.130': 'neck.top_down_layers.0.e_elan_blocks.1.blocks.1',
    'model.131': 'neck.top_down_layers.0.e_elan_blocks.1.blocks.2',
    'model.132': 'neck.top_down_layers.0.e_elan_blocks.1.blocks.3',
    'model.133': 'neck.top_down_layers.0.e_elan_blocks.1.blocks.4',
    'model.134': 'neck.top_down_layers.0.e_elan_blocks.1.blocks.5',
    'model.136': 'neck.top_down_layers.0.e_elan_blocks.1.final_conv',
    'model.138': 'neck.upsample_layers.1.0',
    'model.140': 'neck.reduce_layers.1',

    # neck E-ELANBlock
    'model.142': 'neck.top_down_layers.1.e_elan_blocks.0.short_conv',
    'model.143': 'neck.top_down_layers.1.e_elan_blocks.0.main_conv',
    'model.144': 'neck.top_down_layers.1.e_elan_blocks.0.blocks.0',
    'model.145': 'neck.top_down_layers.1.e_elan_blocks.0.blocks.1',
    'model.146': 'neck.top_down_layers.1.e_elan_blocks.0.blocks.2',
    'model.147': 'neck.top_down_layers.1.e_elan_blocks.0.blocks.3',
    'model.148': 'neck.top_down_layers.1.e_elan_blocks.0.blocks.4',
    'model.149': 'neck.top_down_layers.1.e_elan_blocks.0.blocks.5',
    'model.151': 'neck.top_down_layers.1.e_elan_blocks.0.final_conv',
    'model.152': 'neck.top_down_layers.1.e_elan_blocks.1.short_conv',
    'model.153': 'neck.top_down_layers.1.e_elan_blocks.1.main_conv',
    'model.154': 'neck.top_down_layers.1.e_elan_blocks.1.blocks.0',
    'model.155': 'neck.top_down_layers.1.e_elan_blocks.1.blocks.1',
    'model.156': 'neck.top_down_layers.1.e_elan_blocks.1.blocks.2',
    'model.157': 'neck.top_down_layers.1.e_elan_blocks.1.blocks.3',
    'model.158': 'neck.top_down_layers.1.e_elan_blocks.1.blocks.4',
    'model.159': 'neck.top_down_layers.1.e_elan_blocks.1.blocks.5',
    'model.161': 'neck.top_down_layers.1.e_elan_blocks.1.final_conv',
    'model.163': 'neck.upsample_layers.2.0',
    'model.165': 'neck.reduce_layers.0',
    'model.167': 'neck.top_down_layers.2.e_elan_blocks.0.short_conv',
    'model.168': 'neck.top_down_layers.2.e_elan_blocks.0.main_conv',
    'model.169': 'neck.top_down_layers.2.e_elan_blocks.0.blocks.0',
    'model.170': 'neck.top_down_layers.2.e_elan_blocks.0.blocks.1',
    'model.171': 'neck.top_down_layers.2.e_elan_blocks.0.blocks.2',
    'model.172': 'neck.top_down_layers.2.e_elan_blocks.0.blocks.3',
    'model.173': 'neck.top_down_layers.2.e_elan_blocks.0.blocks.4',
    'model.174': 'neck.top_down_layers.2.e_elan_blocks.0.blocks.5',
    'model.176': 'neck.top_down_layers.2.e_elan_blocks.0.final_conv',
    'model.177': 'neck.top_down_layers.2.e_elan_blocks.1.short_conv',
    'model.178': 'neck.top_down_layers.2.e_elan_blocks.1.main_conv',
    'model.179': 'neck.top_down_layers.2.e_elan_blocks.1.blocks.0',
    'model.180': 'neck.top_down_layers.2.e_elan_blocks.1.blocks.1',
    'model.181': 'neck.top_down_layers.2.e_elan_blocks.1.blocks.2',
    'model.182': 'neck.top_down_layers.2.e_elan_blocks.1.blocks.3',
    'model.183': 'neck.top_down_layers.2.e_elan_blocks.1.blocks.4',
    'model.184': 'neck.top_down_layers.2.e_elan_blocks.1.blocks.5',
    'model.186': 'neck.top_down_layers.2.e_elan_blocks.1.final_conv',
    'model.188.cv1': 'neck.downsample_layers.0.stride_conv_branches.0',
    'model.188.cv2': 'neck.downsample_layers.0.stride_conv_branches.1',
    'model.188.cv3': 'neck.downsample_layers.0.maxpool_branches.1',

    # neck E-ELANBlock
    'model.190': 'neck.bottom_up_layers.0.e_elan_blocks.0.short_conv',
    'model.191': 'neck.bottom_up_layers.0.e_elan_blocks.0.main_conv',
    'model.192': 'neck.bottom_up_layers.0.e_elan_blocks.0.blocks.0',
    'model.193': 'neck.bottom_up_layers.0.e_elan_blocks.0.blocks.1',
    'model.194': 'neck.bottom_up_layers.0.e_elan_blocks.0.blocks.2',
    'model.195': 'neck.bottom_up_layers.0.e_elan_blocks.0.blocks.3',
    'model.196': 'neck.bottom_up_layers.0.e_elan_blocks.0.blocks.4',
    'model.197': 'neck.bottom_up_layers.0.e_elan_blocks.0.blocks.5',
    'model.199': 'neck.bottom_up_layers.0.e_elan_blocks.0.final_conv',
    'model.200': 'neck.bottom_up_layers.0.e_elan_blocks.1.short_conv',
    'model.201': 'neck.bottom_up_layers.0.e_elan_blocks.1.main_conv',
    'model.202': 'neck.bottom_up_layers.0.e_elan_blocks.1.blocks.0',
    'model.203': 'neck.bottom_up_layers.0.e_elan_blocks.1.blocks.1',
    'model.204': 'neck.bottom_up_layers.0.e_elan_blocks.1.blocks.2',
    'model.205': 'neck.bottom_up_layers.0.e_elan_blocks.1.blocks.3',
    'model.206': 'neck.bottom_up_layers.0.e_elan_blocks.1.blocks.4',
    'model.207': 'neck.bottom_up_layers.0.e_elan_blocks.1.blocks.5',
    'model.209': 'neck.bottom_up_layers.0.e_elan_blocks.1.final_conv',
    'model.211.cv1': 'neck.downsample_layers.1.stride_conv_branches.0',
    'model.211.cv2': 'neck.downsample_layers.1.stride_conv_branches.1',
    'model.211.cv3': 'neck.downsample_layers.1.maxpool_branches.1',
    'model.213': 'neck.bottom_up_layers.1.e_elan_blocks.0.short_conv',
    'model.214': 'neck.bottom_up_layers.1.e_elan_blocks.0.main_conv',
    'model.215': 'neck.bottom_up_layers.1.e_elan_blocks.0.blocks.0',
    'model.216': 'neck.bottom_up_layers.1.e_elan_blocks.0.blocks.1',
    'model.217': 'neck.bottom_up_layers.1.e_elan_blocks.0.blocks.2',
    'model.218': 'neck.bottom_up_layers.1.e_elan_blocks.0.blocks.3',
    'model.219': 'neck.bottom_up_layers.1.e_elan_blocks.0.blocks.4',
    'model.220': 'neck.bottom_up_layers.1.e_elan_blocks.0.blocks.5',
    'model.222': 'neck.bottom_up_layers.1.e_elan_blocks.0.final_conv',
    'model.223': 'neck.bottom_up_layers.1.e_elan_blocks.1.short_conv',
    'model.224': 'neck.bottom_up_layers.1.e_elan_blocks.1.main_conv',
    'model.225': 'neck.bottom_up_layers.1.e_elan_blocks.1.blocks.0',
    'model.226': 'neck.bottom_up_layers.1.e_elan_blocks.1.blocks.1',
    'model.227': 'neck.bottom_up_layers.1.e_elan_blocks.1.blocks.2',
    'model.228': 'neck.bottom_up_layers.1.e_elan_blocks.1.blocks.3',
    'model.229': 'neck.bottom_up_layers.1.e_elan_blocks.1.blocks.4',
    'model.230': 'neck.bottom_up_layers.1.e_elan_blocks.1.blocks.5',
    'model.232': 'neck.bottom_up_layers.1.e_elan_blocks.1.final_conv',
    'model.234.cv1': 'neck.downsample_layers.2.stride_conv_branches.0',
    'model.234.cv2': 'neck.downsample_layers.2.stride_conv_branches.1',
    'model.234.cv3': 'neck.downsample_layers.2.maxpool_branches.1',

    # neck E-ELANBlock
    'model.236': 'neck.bottom_up_layers.2.e_elan_blocks.0.short_conv',
    'model.237': 'neck.bottom_up_layers.2.e_elan_blocks.0.main_conv',
    'model.238': 'neck.bottom_up_layers.2.e_elan_blocks.0.blocks.0',
    'model.239': 'neck.bottom_up_layers.2.e_elan_blocks.0.blocks.1',
    'model.240': 'neck.bottom_up_layers.2.e_elan_blocks.0.blocks.2',
    'model.241': 'neck.bottom_up_layers.2.e_elan_blocks.0.blocks.3',
    'model.242': 'neck.bottom_up_layers.2.e_elan_blocks.0.blocks.4',
    'model.243': 'neck.bottom_up_layers.2.e_elan_blocks.0.blocks.5',
    'model.245': 'neck.bottom_up_layers.2.e_elan_blocks.0.final_conv',
    'model.246': 'neck.bottom_up_layers.2.e_elan_blocks.1.short_conv',
    'model.247': 'neck.bottom_up_layers.2.e_elan_blocks.1.main_conv',
    'model.248': 'neck.bottom_up_layers.2.e_elan_blocks.1.blocks.0',
    'model.249': 'neck.bottom_up_layers.2.e_elan_blocks.1.blocks.1',
    'model.250': 'neck.bottom_up_layers.2.e_elan_blocks.1.blocks.2',
    'model.251': 'neck.bottom_up_layers.2.e_elan_blocks.1.blocks.3',
    'model.252': 'neck.bottom_up_layers.2.e_elan_blocks.1.blocks.4',
    'model.253': 'neck.bottom_up_layers.2.e_elan_blocks.1.blocks.5',
    'model.255': 'neck.bottom_up_layers.2.e_elan_blocks.1.final_conv',
    'model.257': 'bbox_head.head_module.main_convs_pred.0.0',
    'model.258': 'bbox_head.head_module.main_convs_pred.1.0',
    'model.259': 'bbox_head.head_module.main_convs_pred.2.0',
    'model.260': 'bbox_head.head_module.main_convs_pred.3.0',

    # head
    'model.261.m.0': 'bbox_head.head_module.main_convs_pred.0.2',
    'model.261.m.1': 'bbox_head.head_module.main_convs_pred.1.2',
    'model.261.m.2': 'bbox_head.head_module.main_convs_pred.2.2',
    'model.261.m.3': 'bbox_head.head_module.main_convs_pred.3.2'
}

convert_dicts = {
    'yolov7-tiny.pt': convert_dict_tiny,
    'yolov7-w6.pt': convert_dict_w,
    'yolov7-e6.pt': convert_dict_e,
    'yolov7-e6e.pt': convert_dict_e2e,
    'yolov7.pt': convert_dict_l,
    'yolov7x.pt': convert_dict_x
}


def convert(src, dst):
    src_key = osp.basename(src)
    convert_dict = convert_dicts[osp.basename(src)]

    num_levels = 3
    if src_key == 'yolov7.pt':
        indexes = [102, 51]
        in_channels = [256, 512, 1024]
    elif src_key == 'yolov7x.pt':
        indexes = [121, 59]
        in_channels = [320, 640, 1280]
    elif src_key == 'yolov7-tiny.pt':
        indexes = [77, 1000]
        in_channels = [128, 256, 512]
    elif src_key == 'yolov7-w6.pt':
        indexes = [118, 47]
        in_channels = [256, 512, 768, 1024]
        num_levels = 4
    elif src_key == 'yolov7-e6.pt':
        indexes = [140, [2, 13, 24, 35, 46, 57, 100, 112, 124]]
        in_channels = 320, 640, 960, 1280
        num_levels = 4
    elif src_key == 'yolov7-e6e.pt':
        indexes = [261, [2, 24, 46, 68, 90, 112, 188, 211, 234]]
        in_channels = 320, 640, 960, 1280
        num_levels = 4

    if isinstance(indexes[1], int):
        indexes[1] = [indexes[1]]
    """Convert keys in detectron pretrained YOLOv7 models to mmyolo style."""
    try:
        yolov7_model = torch.load(src)['model'].float()
        blobs = yolov7_model.state_dict()
    except ModuleNotFoundError:
        raise RuntimeError(
            'This script must be placed under the WongKinYiu/yolov7 repo,'
            ' because loading the official pretrained model need'
            ' `model.py` to build model.')
    state_dict = OrderedDict()

    for key, weight in blobs.items():
        if key.find('anchors') >= 0 or key.find('anchor_grid') >= 0:
            continue

        num, module = key.split('.')[1:3]
        if int(num) < indexes[0] and int(num) not in indexes[1]:
            prefix = f'model.{num}'
            new_key = key.replace(prefix, convert_dict[prefix])
            state_dict[new_key] = weight
            print(f'Convert {key} to {new_key}')
        elif int(num) in indexes[1]:
            strs_key = key.split('.')[:3]
            new_key = key.replace('.'.join(strs_key),
                                  convert_dict['.'.join(strs_key)])
            state_dict[new_key] = weight
            print(f'Convert {key} to {new_key}')
        else:
            strs_key = key.split('.')[:4]
            new_key = key.replace('.'.join(strs_key),
                                  convert_dict['.'.join(strs_key)])
            state_dict[new_key] = weight
            print(f'Convert {key} to {new_key}')

    # Add ImplicitA and ImplicitM
    for i in range(num_levels):
        if num_levels == 3:
            implicit_a = f'bbox_head.head_module.' \
                         f'convs_pred.{i}.0.implicit'
            state_dict[implicit_a] = torch.zeros((1, in_channels[i], 1, 1))
            implicit_m = f'bbox_head.head_module.' \
                         f'convs_pred.{i}.2.implicit'
            state_dict[implicit_m] = torch.ones((1, 3 * 85, 1, 1))
        else:
            implicit_a = f'bbox_head.head_module.' \
                         f'main_convs_pred.{i}.1.implicit'
            state_dict[implicit_a] = torch.zeros((1, in_channels[i], 1, 1))
            implicit_m = f'bbox_head.head_module.' \
                         f'main_convs_pred.{i}.3.implicit'
            state_dict[implicit_m] = torch.ones((1, 3 * 85, 1, 1))

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


# Note: This script must be placed under the yolov7 repo to run.
def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument(
        'src', default='yolov7.pt', help='src yolov7 model path')
    parser.add_argument('dst', default='mm_yolov7l.pt', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)
    print('If your model weights are from P6 models, such as W6, E6, D6, \
            E6E, the auxiliary training module is not required to be loaded, \
            so it is normal for the weights of the auxiliary module \
            to be missing.')


if __name__ == '__main__':
    main()
