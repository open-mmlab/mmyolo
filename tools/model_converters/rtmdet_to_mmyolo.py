# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch


def convert(src, dst):
    """Convert keys in pretrained RTMDet models to MMYOLO style."""
    blobs = torch.load(src)['state_dict']
    state_dict = OrderedDict()

    for key, weight in blobs.items():
        if 'neck.reduce_layers.0' in key:
            new_key = key.replace('.0', '.2')
            state_dict[new_key] = weight
        elif 'neck.reduce_layers.1' in key:
            new_key = key.replace('reduce_layers.1', 'top_down_layers.0.1')
            state_dict[new_key] = weight
        elif 'neck.top_down_blocks.0' in key:
            new_key = key.replace('down_blocks', 'down_layers.0')
            state_dict[new_key] = weight
        elif 'neck.top_down_blocks.1' in key:
            new_key = key.replace('down_blocks', 'down_layers')
            state_dict[new_key] = weight
        elif 'downsamples' in key:
            new_key = key.replace('downsamples', 'downsample_layers')
            state_dict[new_key] = weight
        elif 'bottom_up_blocks' in key:
            new_key = key.replace('bottom_up_blocks', 'bottom_up_layers')
            state_dict[new_key] = weight
        elif 'out_convs' in key:
            new_key = key.replace('out_convs', 'out_layers')
            state_dict[new_key] = weight
        elif 'bbox_head' in key:
            new_key = key.replace('bbox_head', 'bbox_head.head_module')
            state_dict[new_key] = weight
        elif 'data_preprocessor' in key:
            continue
        else:
            new_key = key
            state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    checkpoint['meta'] = blobs.get('meta')
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src rtm model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
