# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from pathlib import Path

import torch
from mmdet.registry import MODELS
from mmengine.analysis import get_model_complexity_info
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmyolo.utils import switch_to_deploy


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[640, 640],
        help='input image size')
    parser.add_argument(
        '--show-arch',
        action='store_true',
        help='whether return the statistics in the form of network layers')
    parser.add_argument(
        '--not-show-table',
        action='store_true',
        help='whether return the statistics in the form of table'),
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    return parser.parse_args()


def inference(args, logger):
    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmyolo'))

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError('invalid input shape')

    # model
    model = MODELS.build(cfg.model)
    if torch.cuda.is_available():
        model.cuda()
    model = revert_sync_batchnorm(model)
    model.eval()
    switch_to_deploy(model)

    # input tensor
    # automatically generate a input tensor with the given input_shape.
    data_batch = {'inputs': [torch.rand(3, h, w)], 'batch_samples': [None]}
    data = model.data_preprocessor(data_batch)
    result = {'ori_shape': (h, w), 'pad_shape': data['inputs'].shape[-2:]}
    outputs = get_model_complexity_info(
        model,
        input_shape=None,
        inputs=data['inputs'],  # the input tensor of the model
        show_table=not args.not_show_table,  # show the complexity table
        show_arch=args.show_arch)  # show the complexity arch

    result['flops'] = outputs['flops_str']
    result['params'] = outputs['params_str']
    result['out_table'] = outputs['out_table']
    result['out_arch'] = outputs['out_arch']

    return result


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)

    split_line = '=' * 30

    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']

    print(result['out_table'])  # print related information by table
    print(result['out_arch'])  # print related information by network layers

    if pad_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {pad_shape}')

    print(f'{split_line}\n'
          f'Input shape: {pad_shape}\nModel Flops: {flops}\n'
          f'Model Parameters: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify '
          'that the flops computation is correct.')


if __name__ == '__main__':
    main()
