# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch

convert_dict_p5 = {
    'model.0': 'backbone.stem',
    'model.1': 'backbone.stage1.0',
    'model.2': 'backbone.stage1.1',
    'model.3': 'backbone.stage2.0',
    'model.4': 'backbone.stage2.1',
    'model.5': 'backbone.stage3.0',
    'model.6': 'backbone.stage3.1',
    'model.7': 'backbone.stage4.0',
    'model.8': 'backbone.stage4.1',
    'model.9.cv1': 'backbone.stage4.2.conv1',
    'model.9.cv2': 'backbone.stage4.2.conv2',
    'model.10': 'neck.reduce_layers.2',
    'model.13': 'neck.top_down_layers.0.0',
    'model.14': 'neck.top_down_layers.0.1',
    'model.17': 'neck.top_down_layers.1',
    'model.18': 'neck.downsample_layers.0',
    'model.20': 'neck.bottom_up_layers.0',
    'model.21': 'neck.downsample_layers.1',
    'model.23': 'neck.bottom_up_layers.1',
    'model.24.m': 'bbox_head.head_module.convs_pred',
    'model.24.proto': 'bbox_head.head_module.proto_preds',
}

convert_dict_p6 = {
    'model.0': 'backbone.stem',
    'model.1': 'backbone.stage1.0',
    'model.2': 'backbone.stage1.1',
    'model.3': 'backbone.stage2.0',
    'model.4': 'backbone.stage2.1',
    'model.5': 'backbone.stage3.0',
    'model.6': 'backbone.stage3.1',
    'model.7': 'backbone.stage4.0',
    'model.8': 'backbone.stage4.1',
    'model.9': 'backbone.stage5.0',
    'model.10': 'backbone.stage5.1',
    'model.11.cv1': 'backbone.stage5.2.conv1',
    'model.11.cv2': 'backbone.stage5.2.conv2',
    'model.12': 'neck.reduce_layers.3',
    'model.15': 'neck.top_down_layers.0.0',
    'model.16': 'neck.top_down_layers.0.1',
    'model.19': 'neck.top_down_layers.1.0',
    'model.20': 'neck.top_down_layers.1.1',
    'model.23': 'neck.top_down_layers.2',
    'model.24': 'neck.downsample_layers.0',
    'model.26': 'neck.bottom_up_layers.0',
    'model.27': 'neck.downsample_layers.1',
    'model.29': 'neck.bottom_up_layers.1',
    'model.30': 'neck.downsample_layers.2',
    'model.32': 'neck.bottom_up_layers.2',
    'model.33.m': 'bbox_head.head_module.convs_pred',
    'model.33.proto': 'bbox_head.head_module.proto_preds',
}


def convert(src, dst):
    """Convert keys in pretrained YOLOv5 models to mmyolo style."""
    if src.endswith('6.pt'):
        convert_dict = convert_dict_p6
        is_p6_model = True
        print('Converting P6 model')
    else:
        convert_dict = convert_dict_p5
        is_p6_model = False
        print('Converting P5 model')
    try:
        yolov5_model = torch.load(src)['model']
        blobs = yolov5_model.state_dict()
    except ModuleNotFoundError:
        raise RuntimeError(
            'This script must be placed under the ultralytics/yolov5 repo,'
            ' because loading the official pretrained model need'
            ' `model.py` to build model.')
    state_dict = OrderedDict()

    for key, weight in blobs.items():

        num, module = key.split('.')[1:3]
        if (is_p6_model and
            (num == '11' or num == '33')) or (not is_p6_model and
                                              (num == '9' or num == '24')):
            if module == 'anchors':
                continue
            prefix = f'model.{num}.{module}'
        else:
            prefix = f'model.{num}'

        new_key = key.replace(prefix, convert_dict[prefix])

        if '.m.' in new_key:
            new_key = new_key.replace('.m.', '.blocks.')
            new_key = new_key.replace('.cv', '.conv')
        elif 'bbox_head.head_module.proto_preds.cv' in new_key:
            new_key = new_key.replace(
                'bbox_head.head_module.proto_preds.cv',
                'bbox_head.head_module.proto_preds.conv')
        else:
            new_key = new_key.replace('.cv1', '.main_conv')
            new_key = new_key.replace('.cv2', '.short_conv')
            new_key = new_key.replace('.cv3', '.final_conv')

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


# Note: This script must be placed under the yolov5 repo to run.
def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument(
        '--src', default='yolov5s.pt', help='src yolov5 model path')
    parser.add_argument('--dst', default='mmyolov5s.pt', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
