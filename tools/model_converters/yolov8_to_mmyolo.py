# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch

convert_dict_s = {
    # backbone
    'model.0': 'backbone.stem',
    'model.1': 'backbone.stage1.0',
    'model.2': 'backbone.stage1.1',
    'model.3': 'backbone.stage2.0',
    'model.4': 'backbone.stage2.1',
    'model.5': 'backbone.stage3.0',
    'model.6': 'backbone.stage3.1',
    'model.7': 'backbone.stage4.0',
    'model.8': 'backbone.stage4.1',
    'model.9': 'backbone.stage4.2',

    # neck
    'model.12': 'neck.top_down_layers.0',
    'model.15': 'neck.top_down_layers.1',
    'model.16': 'neck.downsample_layers.0',
    'model.18': 'neck.bottom_up_layers.0',
    'model.19': 'neck.downsample_layers.1',
    'model.21': 'neck.bottom_up_layers.1',

    # Detector
    'model.22': 'bbox_head.head_module',
}


def convert(src, dst):
    """Convert keys in pretrained YOLOv8 models to mmyolo style."""
    convert_dict = convert_dict_s

    try:
        yolov8_model = torch.load(src)['model']
        blobs = yolov8_model.state_dict()
    except ModuleNotFoundError:
        raise RuntimeError(
            'This script must be placed under the ultralytics repo,'
            ' because loading the official pretrained model need'
            ' `model.py` to build model.'
            'Also need to install hydra-core>=1.2.0 and thop>=0.1.1')
    state_dict = OrderedDict()

    for key, weight in blobs.items():
        num, module = key.split('.')[1:3]
        prefix = f'model.{num}'
        new_key = key.replace(prefix, convert_dict[prefix])

        if '.m.' in new_key:
            new_key = new_key.replace('.m.', '.blocks.')
            new_key = new_key.replace('.cv', '.conv')
        elif 'bbox_head.head_module.proto.cv' in new_key:
            new_key = new_key.replace(
                'bbox_head.head_module.proto.cv',
                'bbox_head.head_module.proto_preds.conv')
        elif 'bbox_head.head_module.proto' in new_key:
            new_key = new_key.replace('bbox_head.head_module.proto',
                                      'bbox_head.head_module.proto_preds')
        elif 'bbox_head.head_module.cv4.' in new_key:
            new_key = new_key.replace(
                'bbox_head.head_module.cv4',
                'bbox_head.head_module.mask_coeff_preds')
            new_key = new_key.replace('.2.weight', '.2.conv.weight')
            new_key = new_key.replace('.2.bias', '.2.conv.bias')
        elif 'bbox_head.head_module' in new_key:
            new_key = new_key.replace('.cv2', '.reg_preds')
            new_key = new_key.replace('.cv3', '.cls_preds')
        elif 'backbone.stage4.2' in new_key:
            new_key = new_key.replace('.cv', '.conv')
        else:
            new_key = new_key.replace('.cv1', '.main_conv')
            new_key = new_key.replace('.cv2', '.final_conv')

        if 'bbox_head.head_module.dfl.conv.weight' == new_key:
            print('Drop "bbox_head.head_module.dfl.conv.weight", '
                  'because it is useless')
            continue
        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict
    torch.save(checkpoint, dst)


# Note: This script must be placed under the ultralytics repo to run.
def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument(
        '--src', default='yolov8s.pt', help='src YOLOv8 model path')
    parser.add_argument('--dst', default='mmyolov8s.pth', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
