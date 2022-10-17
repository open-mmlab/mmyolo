# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import matplotlib.pyplot as plt
from mmengine.config import Config
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Distribution of categories and bbox instances')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--ann-file', default='train', help='dataset ann-file')
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
    dataset_type = cfg.dataset_type
    if args.ann_file == 'train':
        ann_file = cfg.train_dataloader.dataset.ann_file
    elif args.ann_file == 'val':
        ann_file = cfg.val_dataloader.dataset.ann_file

    labels = []
    Num = []
    coco = COCO(osp.join(data_root, ann_file))
    categories = coco.loadCats(coco.getCatIds())
    category_names = [category['name'] for category in categories]

    for category_name in category_names:
        category = coco.getCatIds(catNms=category_name)
        annId = coco.getAnnIds(catIds=category)
        labels.append(category_name)
        Num.append(len(annId))

    fig = plt.figure()
    plt.bar(labels, Num, align='center')
    plt.xticks(rotation=70)
    # X,Y name and title
    plt.xlabel('Category')
    plt.ylabel('Number')
    plt.title(dataset_type)
    out = args.out_dir
    fig.set_size_inches(35, 18)
    fig.savefig(f'{out}/{dataset_type}.jpg')  # Save Image
    # plt.show()  # Show Image


if __name__ == '__main__':
    main()
