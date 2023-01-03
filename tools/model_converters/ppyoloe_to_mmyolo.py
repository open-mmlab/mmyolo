import argparse
import pickle
from collections import OrderedDict

import torch


def convert_bn(k: str):
    name = k.replace('._mean',
                     '.running_mean').replace('._variance', '.running_var')
    return name


def convert_repvgg(k: str):
    if '.conv2.conv1.' in k:
        name = k.replace('.conv2.conv1.', '.conv2.rbr_dense.')
        return name
    elif '.conv2.conv2.' in k:
        name = k.replace('.conv2.conv2.', '.conv2.rbr_1x1.')
        return name
    else:
        return k


def convert(src: str, dst: str, imagenet_pretrain: bool = False):
    with open(src, 'rb') as f:
        model = pickle.load(f)

    new_state_dict = OrderedDict()
    if imagenet_pretrain:
        for k, v in model.items():
            if '@@' in k:
                continue
            if 'stem.' in k:
                # backbone.stem.conv1.conv.weight
                # -> backbone.stem.0.conv.weight
                org_ind = k.split('.')[1][-1]
                new_ind = str(int(org_ind) - 1)
                name = k.replace('stem.conv%s.' % org_ind,
                                 'stem.%s.' % new_ind)
            else:
                # backbone.stages.1.conv2.bn._variance
                # -> backbone.stage2.0.conv2.bn.running_var
                org_stage_ind = k.split('.')[1]
                new_stage_ind = str(int(org_stage_ind) + 1)
                name = k.replace('stages.%s.' % org_stage_ind,
                                 'stage%s.0.' % new_stage_ind)
                name = convert_repvgg(name)
                if '.attn.' in k:
                    name = name.replace('.attn.fc.', '.attn.fc.conv.')
            name = convert_bn(name)
            name = 'backbone.' + name

            new_state_dict[name] = torch.from_numpy(v)
    else:
        for k, v in model.items():
            name = k
            if k.startswith('backbone.'):
                if '.stem.' in k:
                    # backbone.stem.conv1.conv.weight
                    # -> backbone.stem.0.conv.weight
                    org_ind = k.split('.')[2][-1]
                    new_ind = str(int(org_ind) - 1)
                    name = k.replace('.stem.conv%s.' % org_ind,
                                     '.stem.%s.' % new_ind)
                else:
                    # backbone.stages.1.conv2.bn._variance
                    # -> backbone.stage2.0.conv2.bn.running_var
                    org_stage_ind = k.split('.')[2]
                    new_stage_ind = str(int(org_stage_ind) + 1)
                    name = k.replace('.stages.%s.' % org_stage_ind,
                                     '.stage%s.0.' % new_stage_ind)
                    name = convert_repvgg(name)
                    if '.attn.' in k:
                        name = name.replace('.attn.fc.', '.attn.fc.conv.')
                name = convert_bn(name)
            elif k.startswith('neck.'):
                # fpn_stages
                if k.startswith('neck.fpn_stages.'):
                    # neck.fpn_stages.0.0.conv1.conv.weight
                    # -> neck.reduce_layers.2.0.conv1.conv.weight
                    if k.startswith('neck.fpn_stages.0.0.'):
                        name = k.replace('neck.fpn_stages.0.0.',
                                         'neck.reduce_layers.2.0.')
                        if '.spp.' in name:
                            name = name.replace('.spp.conv.', '.spp.conv2.')
                    # neck.fpn_stages.1.0.conv1.conv.weight
                    # -> neck.top_down_layers.0.0.conv1.conv.weight
                    elif k.startswith('neck.fpn_stages.1.0.'):
                        name = k.replace('neck.fpn_stages.1.0.',
                                         'neck.top_down_layers.0.0.')
                    elif k.startswith('neck.fpn_stages.2.0.'):
                        name = k.replace('neck.fpn_stages.2.0.',
                                         'neck.top_down_layers.1.0.')
                    else:
                        raise NotImplementedError('Not implemented.')
                    name = name.replace('.0.convs.', '.0.blocks.')
                elif k.startswith('neck.fpn_routes.'):
                    # neck.fpn_routes.0.conv.weight
                    # -> neck.upsample_layers.0.0.conv.weight
                    index = k.split('.')[2]
                    name = 'neck.upsample_layers.' + index + '.0.' + '.'.join(
                        k.split('.')[-2:])
                    name = name.replace('.0.convs.', '.0.blocks.')
                elif k.startswith('neck.pan_stages.'):
                    # neck.pan_stages.0.0.conv1.conv.weight
                    # -> neck.bottom_up_layers.1.0.conv1.conv.weight
                    ind = k.split('.')[2]
                    name = k.replace(
                        'neck.pan_stages.' + ind, 'neck.bottom_up_layers.' +
                        ('0' if ind == '1' else '1'))
                    name = name.replace('.0.convs.', '.0.blocks.')
                elif k.startswith('neck.pan_routes.'):
                    # neck.pan_routes.0.conv.weight
                    # -> neck.downsample_layers.0.conv.weight
                    ind = k.split('.')[2]
                    name = k.replace(
                        'neck.pan_routes.' + ind, 'neck.downsample_layers.' +
                        ('0' if ind == '1' else '1'))
                    name = name.replace('.0.convs.', '.0.blocks.')

                else:
                    raise NotImplementedError('Not implement.')
                name = convert_repvgg(name)
                name = convert_bn(name)
            elif k.startswith('yolo_head.'):
                if ('anchor_points' in k) or ('stride_tensor' in k):
                    continue
                if 'proj_conv' in k:
                    name = k.replace('yolo_head.proj_conv.',
                                     'bbox_head.head_module.proj_conv.')
                else:
                    for org_key, rep_key in [
                        [
                            'yolo_head.stem_cls.',
                            'bbox_head.head_module.cls_stems.'
                        ],
                        [
                            'yolo_head.stem_reg.',
                            'bbox_head.head_module.reg_stems.'
                        ],
                        [
                            'yolo_head.pred_cls.',
                            'bbox_head.head_module.cls_preds.'
                        ],
                        [
                            'yolo_head.pred_reg.',
                            'bbox_head.head_module.reg_preds.'
                        ]
                    ]:
                        name = name.replace(org_key, rep_key)
                    name = name.split('.')
                    ind = name[3]
                    name[3] = str(2 - int(ind))
                    name = '.'.join(name)
                name = convert_bn(name)
            else:
                continue

            new_state_dict[name] = torch.from_numpy(v)
    data = {'state_dict': new_state_dict}
    torch.save(data, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument(
        '--src',
        default='ppyoloe_plus_crn_s_80e_coco.pdparams',
        help='src ppyoloe model path')
    parser.add_argument(
        '--dst', default='mmppyoloe_plus_s.pt', help='save path')
    parser.add_argument(
        '--imagenet-pretrain',
        action='store_true',
        default=False,
        help='Load model pretrained on imagenet dataset which only '
        'have weight for backbone.')
    args = parser.parse_args()
    convert(args.src, args.dst, args.imagenet_pretrain)


if __name__ == '__main__':
    main()
