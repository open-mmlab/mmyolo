import argparse
from collections import OrderedDict

import torch


def convert(src, dst):
    import sys
    sys.path.append('yolov6')
    try:
        ckpt = torch.load(src, map_location=torch.device('cpu'))
    except ModuleNotFoundError:
        raise RuntimeError(
            'This script must be placed under the meituan/YOLOv6 repo,'
            ' because loading the official pretrained model need'
            ' some python files to build model.')
    # The saved model is the model before reparameterization
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        name = k
        if 'detect' in k:
            if 'proj' in k:
                continue
            name = k.replace('detect', 'bbox_head.head_module')
        if k.find('anchors') >= 0 or k.find('anchor_grid') >= 0:
            continue

        if 'ERBlock_2' in k:
            name = k.replace('ERBlock_2', 'stage1.0')
            if '.cv' in k:
                name = name.replace('.cv', '.conv')
            if '.m.' in k:
                name = name.replace('.m.', '.block.')
        elif 'ERBlock_3' in k:
            name = k.replace('ERBlock_3', 'stage2.0')
            if '.cv' in k:
                name = name.replace('.cv', '.conv')
            if '.m.' in k:
                name = name.replace('.m.', '.block.')
        elif 'ERBlock_4' in k:
            name = k.replace('ERBlock_4', 'stage3.0')
            if '.cv' in k:
                name = name.replace('.cv', '.conv')
            if '.m.' in k:
                name = name.replace('.m.', '.block.')
        elif 'ERBlock_5' in k:
            name = k.replace('ERBlock_5', 'stage4.0')
            if '.cv' in k:
                name = name.replace('.cv', '.conv')
            if '.m.' in k:
                name = name.replace('.m.', '.block.')
            if 'stage4.0.2' in name:
                name = name.replace('stage4.0.2', 'stage4.1')
                name = name.replace('cv', 'conv')
        elif 'reduce_layer0' in k:
            name = k.replace('reduce_layer0', 'reduce_layers.2')
        elif 'Rep_p4' in k:
            name = k.replace('Rep_p4', 'top_down_layers.0.0')
            if '.cv' in k:
                name = name.replace('.cv', '.conv')
            if '.m.' in k:
                name = name.replace('.m.', '.block.')
        elif 'reduce_layer1' in k:
            name = k.replace('reduce_layer1', 'top_down_layers.0.1')
            if '.cv' in k:
                name = name.replace('.cv', '.conv')
            if '.m.' in k:
                name = name.replace('.m.', '.block.')
        elif 'Rep_p3' in k:
            name = k.replace('Rep_p3', 'top_down_layers.1')
            if '.cv' in k:
                name = name.replace('.cv', '.conv')
            if '.m.' in k:
                name = name.replace('.m.', '.block.')
        elif 'upsample0' in k:
            name = k.replace('upsample0.upsample_transpose',
                             'upsample_layers.0')
        elif 'upsample1' in k:
            name = k.replace('upsample1.upsample_transpose',
                             'upsample_layers.1')
        elif 'Rep_n3' in k:
            name = k.replace('Rep_n3', 'bottom_up_layers.0')
            if '.cv' in k:
                name = name.replace('.cv', '.conv')
            if '.m.' in k:
                name = name.replace('.m.', '.block.')
        elif 'Rep_n4' in k:
            name = k.replace('Rep_n4', 'bottom_up_layers.1')
            if '.cv' in k:
                name = name.replace('.cv', '.conv')
            if '.m.' in k:
                name = name.replace('.m.', '.block.')
        elif 'downsample2' in k:
            name = k.replace('downsample2', 'downsample_layers.0')
        elif 'downsample1' in k:
            name = k.replace('downsample1', 'downsample_layers.1')

        new_state_dict[name] = v
    data = {'state_dict': new_state_dict}
    torch.save(data, dst)


# Note: This script must be placed under the yolov6 repo to run.
def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument(
        '--src', default='yolov6s.pt', help='src yolov6 model path')
    parser.add_argument('--dst', default='mmyolov6.pt', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
