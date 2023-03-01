# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import subprocess

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')

    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    if 'message_hub' in checkpoint:
        del checkpoint['message_hub']
    if 'ema_state_dict' in checkpoint:
        del checkpoint['ema_state_dict']

    for key in list(checkpoint['state_dict']):
        if key.startswith('data_preprocessor'):
            checkpoint['state_dict'].pop(key)
        elif 'priors_base_sizes' in key:
            checkpoint['state_dict'].pop(key)
        elif 'grid_offset' in key:
            checkpoint['state_dict'].pop(key)
        elif 'prior_inds' in key:
            checkpoint['state_dict'].pop(key)

    if torch.__version__ >= '1.6':
        torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file
    final_file = out_file_name + f'-{sha[:8]}.pth'
    subprocess.Popen(['mv', out_file, final_file])


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
