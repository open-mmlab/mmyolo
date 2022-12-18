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
def _process_data(args,
                  in_dataset_type: str,
                  out_dataset_type: str,
                  year: str = '2017'):
    assert in_dataset_type in ('train', 'val')
    assert out_dataset_type in ('train', 'val')

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

    area_dict = {
        'small': [0., 32 * 32],
        'medium': [32 * 32, 96 * 96],
        'large': [96 * 96, float('inf')]
    }

    coco = COCO(ann_path)

    # filter annotations by category ids and area range
    areaRng = area_dict[args.area_size] if args.area_size else []
    catIds = coco.getCatIds(args.classes) if args.classes else []
    ann_ids = coco.getAnnIds(catIds=catIds, areaRng=areaRng)
    ann_info = coco.loadAnns(ann_ids)

    # get image ids by anns set
    filter_img_ids = {ann['image_id'] for ann in ann_info}
    filter_img = coco.loadImgs(filter_img_ids)

    # shuffle
    np.random.shuffle(filter_img)

    num_img = args.num_img if args.num_img > 0 else len(filter_img)
    if num_img > len(filter_img):
        print(
            f'num_img is too big, will be set to {len(filter_img)}, '
            'because of not enough image after filter by classes and area_size'
        )
        num_img = len(filter_img)

    progress_bar = mmengine.ProgressBar(num_img)

    for i in range(num_img):
        file_name = filter_img[i]['file_name']
        image_path = osp.join(args.root, in_dataset_type + year, file_name)

        ann_ids = coco.getAnnIds(
            imgIds=[filter_img[i]['id']], catIds=catIds, areaRng=areaRng)
        img_ann_info = coco.loadAnns(ann_ids)

        new_json_data['images'].append(filter_img[i])
        new_json_data['annotations'].extend(img_ann_info)

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
        '--num-img',
        default=50,
        type=int,
        help='num of extract image, -1 means all images')
    parser.add_argument(
        '--area-size',
        choices=['small', 'medium', 'large'],
        help='filter ground-truth info by area size')
    parser.add_argument(
        '--classes', nargs='+', help='filter ground-truth by class name')
    parser.add_argument(
        '--use-training-set',
        action='store_true',
        help='Whether to use the training set when extract the training set. '
        'The training subset is extracted from the validation set by '
        'default which can speed up.')
    parser.add_argument('--seed', default=-1, type=int, help='seed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out_dir != args.root, \
        'The file will be overwritten in place, ' \
        'so the same folder is not allowed !'

    seed = int(args.seed)
    if seed != -1:
        print(f'Set the global seed: {seed}')
        np.random.seed(int(args.seed))

    _make_dirs(args.out_dir)

    print('====Start processing train dataset====')
    if args.use_training_set:
        _process_data(args, 'train', 'train')
    else:
        _process_data(args, 'val', 'train')
    print('\n====Start processing val dataset====')
    _process_data(args, 'val', 'val')
    print(f'\n Result save to {args.out_dir}')


if __name__ == '__main__':
    main()
