# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import matplotlib.pyplot as plt
from mmengine.config import Config
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Proportional distribution of bbox width and height')
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
    if args.ann_file == 'train':
        ann_file = cfg.train_dataloader.dataset.ann_file
    elif args.ann_file == 'val':
        ann_file = cfg.val_dataloader.dataset.ann_file

    labels = []
    ann_bbox_w = []
    ann_bbow_h = []
    ratio = []
    count_list = list()
    ratio_sum = []
    ratio_list = []
    coco = COCO(osp.join(data_root, ann_file))
    categories = coco.loadCats(coco.getCatIds())
    category_names = [category['name'] for category in categories]

    for category_name in category_names:
        category = coco.getCatIds(catNms=category_name)
        annId = coco.getAnnIds(catIds=category)
        labels.append(category_name)
        annotations = coco.loadAnns(annId)
        anns = [ann['bbox'] for ann in annotations]
        for i in range(len(anns)):
            ann = anns[i]
            bbox_w = ann[2]
            bbow_h = ann[3]
            ann_bbox_w.append(bbox_w)
            ann_bbow_h.append(bbow_h)
            ratios = bbox_w / bbow_h
            ratio.append(round(ratios, 2))
        count_set = set(ratio)
        for t in count_set:
            count_list.append((t, ratio.count(t)))
        for n in range(len(count_list)):
            count = count_list[n]
            ratio_sums = count[1]
            ratio_lists = count[0]
            ratio_sum.append(ratio_sums)
            ratio_list.append(ratio_lists)

        fig = plt.figure()
        plt.scatter(ratio_list, ratio_sum)
        plt.xlabel('The width to height ratio')
        plt.ylabel('Quantity of same width to height ratio')
        plt.title(category_name)
        out = args.out_dir
        fig.set_size_inches(35, 18)
        plt.savefig(f'{out}/{category_name}.jpg')  # Save Image
        # plt.show()  # Show Image


if __name__ == '__main__':
    main()
