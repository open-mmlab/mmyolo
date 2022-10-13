# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch

convert_dict = {
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
    'model.13': 'backbone.stage2.0.top_branches.1',
    'model.14': 'backbone.stage2.0.bottom_branches.0',
    'model.15': 'backbone.stage2.0.bottom_branches.1',
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
    'model.26': 'backbone.stage3.0.top_branches.1',
    'model.27': 'backbone.stage3.0.bottom_branches.0',
    'model.28': 'backbone.stage3.0.bottom_branches.1',
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
    'model.39': 'backbone.stage4.0.top_branches.1',
    'model.40': 'backbone.stage4.0.bottom_branches.0',
    'model.41': 'backbone.stage4.0.bottom_branches.1',
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
    'model.51.cv2': 'neck.reduce_layers.2.short_layers',
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
    'model.77': 'neck.downsample_layers.0.top_branches.1',
    'model.78': 'neck.downsample_layers.0.bottom_branches.0',
    'model.79': 'neck.downsample_layers.0.bottom_branches.1',

    # neck ELANBlock reduce_channel_2x
    'model.81': 'neck.bottom_up_layers.0.short_conv',
    'model.82': 'neck.bottom_up_layers.0.main_conv',
    'model.83': 'neck.bottom_up_layers.0.blocks.0',
    'model.84': 'neck.bottom_up_layers.0.blocks.1',
    'model.85': 'neck.bottom_up_layers.0.blocks.2',
    'model.86': 'neck.bottom_up_layers.0.blocks.3',
    'model.88': 'neck.bottom_up_layers.0.final_conv',

    # neck MaxPoolBlock no_change_channel
    'model.90': 'neck.downsample_layers.1.top_branches.1',
    'model.91': 'neck.downsample_layers.1.bottom_branches.0',
    'model.92': 'neck.downsample_layers.1.bottom_branches.1',

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
    'model.105.m': 'bbox_head.head_module.convs_pred'
}


def convert(src, dst):
    """Convert keys in detectron pretrained YOLOv7 models to mmyolo style."""
    yolov7_model = torch.load(src)['model'].float()
    blobs = yolov7_model.state_dict()
    state_dict = OrderedDict()

    for key, weight in blobs.items():
        if key.find('anchors') >= 0 or key.find('anchor_grid') >= 0:
            continue

        num, module = key.split('.')[1:3]
        if int(num) < 102 and int(num) != 51:
            prefix = f'model.{num}'
            new_key = key.replace(prefix, convert_dict[prefix])
            state_dict[new_key] = weight
            print(f'Convert {key} to {new_key}')
        elif int(num) < 105 and int(num) != 51:
            strs_key = key.split('.')[:4]
            new_key = key.replace('.'.join(strs_key),
                                  convert_dict['.'.join(strs_key)])
            state_dict[new_key] = weight
            print(f'Convert {key} to {new_key}')
        else:
            strs_key = key.split('.')[:3]
            new_key = key.replace('.'.join(strs_key),
                                  convert_dict['.'.join(strs_key)])
            state_dict[new_key] = weight
            print(f'Convert {key} to {new_key}')

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


# Note: This script must be placed under the yolov7 repo to run.
def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument(
        '--src', default='yolov7.pt', help='src yolov7 model path')
    parser.add_argument('--dst', default='mm_yolov7l.pt', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
