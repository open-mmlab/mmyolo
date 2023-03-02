# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path

import torch


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
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
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
        torch.save(checkpoint, args.checkpoint)
    else:
        ckpt_path = Path(args.checkpoint)
        ckpt_name = ckpt_path.stem
        if args.out_path:
            ckpt_dir = Path(args.out_path)
        else:
            ckpt_dir = ckpt_path.parent
        new_ckpt_path = ckpt_dir / f'{ckpt_name}_student.pth'
        torch.save(checkpoint, new_ckpt_path)


if __name__ == '__main__':
    main()
