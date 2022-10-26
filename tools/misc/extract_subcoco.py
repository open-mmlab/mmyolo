# Copyright (c) OpenMMLab. All rights reserved.
"""Extracting subsets from coco2017 dataset.

This script is mainly used to debug and verify the correctness of the
program quickly.
The root folder format must be in the following format:

├── root
│   ├── annotations
│   ├── train2017
│   ├── val2017
│   ├── test2017

Currently, only support COCO2017. In the future will support user-defined
datasets of standard coco JSON format.

Example:
   python tools/misc/extract_subcoco.py ${ROOT} ${OUT_DIR} --num-img ${NUM_IMG}
"""

import argparse
import os.path as osp
import shutil

import mmengine
import numpy as np
from pycocotools.coco import COCO


# TODO: Currently only supports coco2017
def _process_data(args, in_dataset_type: str, out_dataset_type: str):
    assert in_dataset_type in ('train', 'val')
    assert out_dataset_type in ('train', 'val')

    year = '2017'

    int_ann_file_name = f'annotations/instances_{in_dataset_type}{year}.json'
    out_ann_file_name = f'annotations/instances_{out_dataset_type}{year}.json'

    ann_path = osp.join(args.root, int_ann_file_name)
    json_data = mmengine.load(ann_path)

    new_json_data = {
        'info': json_data['info'],
        'licenses': json_data['licenses'],
        'categories': json_data['categories'],
        'images': [],
        'annotations': []
    }

    images = json_data['images']
    coco = COCO(ann_path)

    # shuffle
    np.random.shuffle(images)

    progress_bar = mmengine.ProgressBar(args.num_img)

    for i in range(args.num_img):
        file_name = images[i]['file_name']
        image_path = osp.join(args.root, in_dataset_type + year, file_name)

        ann_ids = coco.getAnnIds(imgIds=[images[i]['id']])
        ann_info = coco.loadAnns(ann_ids)

        new_json_data['images'].append(images[i])
        new_json_data['annotations'].extend(ann_info)

        shutil.copy(image_path, osp.join(args.out_dir,
                                         out_dataset_type + year))

        progress_bar.update()

    mmengine.dump(new_json_data, osp.join(args.out_dir, out_ann_file_name))


def _make_dirs(out_dir):
    mmengine.mkdir_or_exist(out_dir)
    mmengine.mkdir_or_exist(osp.join(out_dir, 'annotations'))
    mmengine.mkdir_or_exist(osp.join(out_dir, 'train2017'))
    mmengine.mkdir_or_exist(osp.join(out_dir, 'val2017'))


def parse_args():
    parser = argparse.ArgumentParser(description='Extract coco subset')
    parser.add_argument('root', help='root path')
    parser.add_argument(
        'out_dir', type=str, help='directory where subset coco will be saved.')
    parser.add_argument(
        '--num-img', default=50, type=int, help='num of extract image')
    parser.add_argument(
        '--use-training-set',
        action='store_true',
        help='Whether to use the training set when extract the training set. '
        'The training subset is extracted from the validation set by '
        'default which can speed up.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out_dir != args.root, \
        'The file will be overwritten in place, ' \
        'so the same folder is not allowed !'

    _make_dirs(args.out_dir)

    print('start processing train dataset')
    if args.use_training_set:
        _process_data(args, 'train', 'train')
    else:
        _process_data(args, 'val', 'train')
    print('start processing val dataset')
    _process_data(args, 'val', 'val')


if __name__ == '__main__':
    main()
