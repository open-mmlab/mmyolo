# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from pathlib import Path

from mmengine.runner import CheckpointLoader, save_checkpoint
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert KD checkpoint to student-only checkpoint')
    parser.add_argument('checkpoint', help='input checkpoint filename')
    parser.add_argument('--out-path', help='save checkpoint path')
    parser.add_argument(
        '--inplace', action='store_true', help='replace origin ckpt')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    checkpoint = CheckpointLoader.load_checkpoint(
        args.checkpoint, map_location='cpu')
    new_state_dict = dict()
    new_meta = checkpoint['meta']

    for key, value in checkpoint['state_dict'].items():
        if key.startswith('architecture.'):
            new_key = key.replace('architecture.', '')
            new_state_dict[new_key] = value

    checkpoint = dict()
    checkpoint['meta'] = new_meta
    checkpoint['state_dict'] = new_state_dict

    if args.inplace:
        assert osp.exists(args.checkpoint), \
            'can not find the checkpoint path: {args.checkpoint}'
        save_checkpoint(checkpoint, args.checkpoint)
    else:
        ckpt_path = Path(args.checkpoint)
        ckpt_name = ckpt_path.stem
        if args.out_path:
            ckpt_dir = Path(args.out_path)
        else:
            ckpt_dir = ckpt_path.parent
        mkdir_or_exist(ckpt_dir)
        new_ckpt_path = osp.join(ckpt_dir, f'{ckpt_name}_student.pth')
        save_checkpoint(checkpoint, new_ckpt_path)


if __name__ == '__main__':
    main()
