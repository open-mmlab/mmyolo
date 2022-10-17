# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import matplotlib.pyplot as plt
from mmengine.config import Config
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Classification and area distribution')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--ann-file', default='train', help='dataset ann-file')
    parser.add_argument('--size', default=None, help='dataset type')
    parser.add_argument(
        '--out-dir',
        default='./data',
        type=str,
        help='If there is no display interface, you can save it')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    data_root = cfg.data_root
    if args.ann_file == 'train':
        ann_file = cfg.train_dataloader.dataset.ann_file
    elif args.ann_file == 'val':
        ann_file = cfg.val_dataloader.dataset.ann_file

    labels = []
    small = []
    small_num = []
    medium = []
    medium_num = []
    large = []
    large_num = []
    coco = COCO(osp.join(data_root, ann_file))
    categories = coco.loadCats(coco.getCatIds())
    category_names = [category['name'] for category in categories]

    for category_name in category_names:
        category = coco.getCatIds(catNms=category_name)
        annId = coco.getAnnIds(catIds=category)
        labels.append(category_name)
        annotations = coco.loadAnns(annId)
        areas = [area['area'] for area in annotations]
        for i in range(len(areas)):
            area = areas[i]
            if area < 1024:
                small.append(area)
            elif 1024 < area < 9216:
                medium.append(area)
            else:
                large.append(area)
        small_num.append(len(small))
        medium_num.append(len(medium))
        large_num.append(len(large))
    figure_name = args.size
    fig = plt.figure()
    if figure_name == 'small':
        plt.bar(labels, small_num, align='center')
    elif figure_name == 'medium':
        plt.bar(labels, medium_num, align='center')
    elif figure_name == 'large':
        plt.bar(labels, large_num, align='center')
    else:
        print('Please enter the correct size, such as small, medium, large')
    plt.xticks(rotation=70)
    # X,Y name and title
    plt.xlabel('Category')
    plt.ylabel('Area')
    plt.title(f'{figure_name}')
    out = args.out_dir
    fig.set_size_inches(35, 18)
    fig.savefig(f'{out}/{figure_name}.jpg')  # Save Image
    # plt.show()  # Show Image


if __name__ == '__main__':
    main()
