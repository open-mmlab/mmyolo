# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch

neck_dict = {
    'backbone.lateral_conv0': 'neck.reduce_layers.2',
    'backbone.C3_p4.conv': 'neck.top_down_layers.0.0.cv',
    'backbone.C3_p4.m.0.': 'neck.top_down_layers.0.0.m.0.',
    'backbone.reduce_conv1': 'neck.top_down_layers.0.1',
    'backbone.C3_p3.conv': 'neck.top_down_layers.1.cv',
    'backbone.C3_p3.m.0.': 'neck.top_down_layers.1.m.0.',
    'backbone.bu_conv2': 'neck.downsample_layers.0',
    'backbone.C3_n3.conv': 'neck.bottom_up_layers.0.cv',
    'backbone.C3_n3.m.0.': 'neck.bottom_up_layers.0.m.0.',
    'backbone.bu_conv1': 'neck.downsample_layers.1',
    'backbone.C3_n4.conv': 'neck.bottom_up_layers.1.cv',
    'backbone.C3_n4.m.0.': 'neck.bottom_up_layers.1.m.0.',
}


def convert_stem(model_key, model_weight, state_dict, converted_names):
    new_key = model_key[9:]
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_backbone(model_key, model_weight, state_dict, converted_names):
    new_key = model_key.replace('backbone.dark', 'stage')
    num = int(new_key[14]) - 1
    new_key = new_key[:14] + str(num) + new_key[15:]
    if '.m.' in model_key:
        new_key = new_key.replace('.m.', '.blocks.')
    elif not new_key[16] == '0' and 'stage4.1' not in new_key:
        new_key = new_key.replace('conv1', 'main_conv')
        new_key = new_key.replace('conv2', 'short_conv')
        new_key = new_key.replace('conv3', 'final_conv')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_neck(model_key, model_weight, state_dict, converted_names):
    for old, new in neck_dict.items():
        if old in model_key:
            new_key = model_key.replace(old, new)
    if '.m.' in model_key:
        new_key = new_key.replace('.m.', '.blocks.')
    elif '.C' in model_key:
        new_key = new_key.replace('cv1', 'main_conv')
        new_key = new_key.replace('cv2', 'short_conv')
        new_key = new_key.replace('cv3', 'final_conv')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert_head(model_key, model_weight, state_dict, converted_names):
    if 'stem' in model_key:
        new_key = model_key.replace('head.stem', 'neck.out_layer')
    elif 'cls_convs' in model_key:
        new_key = model_key.replace(
            'head.cls_convs', 'bbox_head.head_module.multi_level_cls_convs')
    elif 'reg_convs' in model_key:
        new_key = model_key.replace(
            'head.reg_convs', 'bbox_head.head_module.multi_level_reg_convs')
    elif 'preds' in model_key:
        new_key = model_key.replace('head.',
                                    'bbox_head.head_module.multi_level_conv_')
        new_key = new_key.replace('_preds', '')
    state_dict[new_key] = model_weight
    converted_names.add(model_key)
    print(f'Convert {model_key} to {new_key}')


def convert(src, dst):
    """Convert keys in detectron pretrained YOLOX models to mmyolo style."""
    blobs = torch.load(src)['model']
    state_dict = OrderedDict()
    converted_names = set()

    for key, weight in blobs.items():
        if 'backbone.stem' in key:
            convert_stem(key, weight, state_dict, converted_names)
        elif 'backbone.backbone' in key:
            convert_backbone(key, weight, state_dict, converted_names)
        elif 'backbone.neck' not in key and 'head' not in key:
            convert_neck(key, weight, state_dict, converted_names)
        elif 'head' in key:
            convert_head(key, weight, state_dict, converted_names)

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument(
        '--src', default='yolox_s.pth', help='src yolox model path')
    parser.add_argument('--dst', default='mmyoloxs.pt', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
