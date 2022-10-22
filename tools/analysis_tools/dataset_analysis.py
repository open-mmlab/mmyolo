# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mmengine.config import Config
from mmengine.utils import ProgressBar

from mmyolo.registry import DATASETS
from mmyolo.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Distribution of categories and bbox instances')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--output-dir',
        default='./',
        type=str,
        help='If there is no display interface, you can save it')
    args = parser.parse_args()
    return args


def show_bbox_num(cfg, classes, cat_nums, out):
    """Display the distribution map of categories and number of bbox
    instances."""

    fig = plt.figure(figsize=(50, 18), dpi=600)
    plt.bar(classes, cat_nums, align='center')
    plt.xticks(rotation=70)
    plt.xlabel('Category Name')
    plt.ylabel('Num of instances')
    plt.title(cfg.dataset_type)
    for x, y in enumerate(cat_nums):
        plt.text(x, y + 10, '%s' % y, ha='center', fontsize=4)
    out_dir = os.path.join(out, 'show_bbox_num')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(f'{out_dir}/{cfg.dataset_type}.jpg')  # Save Image


def show_bbox_wh(which_cat_w, which_cat_h, classes, idx, out):
    """Display the width and height distribution of categories and bbox
    instances."""

    fig = plt.figure(figsize=(30, 18), dpi=80)
    plt.scatter(which_cat_w, which_cat_h, s=10)
    plt.xlabel('Width of bbox')
    plt.ylabel('Height of bbox')
    plt.title(f'Current Display Category:{classes[idx]}')
    out_dir = os.path.join(out, 'show_bbox_wh')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(f'{out_dir}/{classes[idx]}.jpg')  # Save Image
    matplotlib.pyplot.close()


def show_bbox_wh_ratio(cat_ratio_nums, classes, idx, out):
    """Display the distribution map of category and bbox instance width and
    height ratio."""

    cat_ratio_num = list(np.concatenate(cat_ratio_nums))
    fig = plt.figure(figsize=(20, 18), dpi=100)
    labels = 'Less than equal', 'Equal', 'More than equal'
    values = cat_ratio_num[0], cat_ratio_num[1], cat_ratio_num[2]
    colors = '#438675', '#F7B469', '#6BA6DA'
    explode = 0, 0, 0
    plt.pie(
        values,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=50)
    plt.axis('equal')
    plt.title(f'Current Display Category:{classes[idx]}')
    out_dir = os.path.join(out, 'show_bbox_wh_ratio')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(f'{out_dir}/{classes[idx]}.jpg')  # Save Image
    matplotlib.pyplot.close()


def show_bbox_area(cfg, classes, cat_area_num, out):
    """Display the distribution map of category and bbox instance area based on
    the rules of large, medium and small objects."""

    area_s_num = [cat_area_num[idx][0] for idx in range(len(classes))]
    area_m_mun = [cat_area_num[idx][1] for idx in range(len(classes))]
    area_l_num = [cat_area_num[idx][2] for idx in range(len(classes))]
    x = np.arange(len(classes))
    width = 0.3
    fig = plt.figure(figsize=(50, 18), dpi=600)
    plt.bar(x - width, area_s_num, width, label='Small', color='#438675')
    plt.bar(x, area_m_mun, width, label='Medium', color='#F7B469')
    plt.bar(x + width, area_l_num, width, label='Large', color='#6BA6DA')
    color = ['#438675', '#F7B469', '#6BA6DA']
    labels = ['Small', 'Medium', 'Large']
    patches = [
        mpatches.Patch(color=color[i], label=f'{labels[i]:s}')
        for i in range(len(color))
    ]
    plt.ylabel('Category Area')
    plt.xlabel('Category Name')
    plt.title(
        'Area and number of large, medium and small objects of each category')
    plt.xticks(x, classes)
    plt.xticks(rotation=70)
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(handles=patches, bbox_to_anchor=(0.55, 1.12), ncol=3)
    for x1, y1 in enumerate(area_s_num):
        plt.text(x1 - width, y1 + 8, y1, ha='center', fontsize=4)
    for x2, y2 in enumerate(area_m_mun):
        plt.text(x2, y2 + 5, y2, ha='center', fontsize=4)
    for x3, y3 in enumerate(area_l_num):
        plt.text(x3 + width, y3 + 8, y3, ha='center', fontsize=4)
    out_dir = os.path.join(out, 'show_bbox_area')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(f'{out_dir}/{cfg.dataset_type}.jpg')  # Save Image


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # register all modules in mmdet into the registries
    register_all_modules()

    # Build Dataset
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    data_list = dataset.load_data_list()
    progress_bar = ProgressBar(len(dataset))

    # Call the category name and save address
    classes = dataset.metainfo['CLASSES']
    out = args.output_dir
    # Build an all 0 array to store the number of each category
    cat_nums = np.zeros((len(classes), ), dtype=int)
    # Build an array to store bbox instance data
    cat_with_bbox = [[] for _ in classes]
    # Build a list to store the data after bbox area division
    cat_area_num = []

    # Get the quantity and bbox data corresponding to each category
    for item in data_list:
        for instance in item['instances']:
            cat_nums[
                instance['bbox_label']] = cat_nums[instance['bbox_label']] + 1
            cat_with_bbox[instance['bbox_label']].append(instance['bbox'])
        progress_bar.update()
    print('\n\nStart drawing')
    show_bbox_num(cfg, classes, cat_nums, out)

    progress_bar_classes = ProgressBar(len(classes))
    # Get the width, height and area of bbox corresponding to each category
    for idx in range(len(classes)):
        cat_bbox = np.array(cat_with_bbox[idx])
        cat_bbox_w = cat_bbox[:, 2] - cat_bbox[:, 0]
        cat_bbox_h = cat_bbox[:, 3] - cat_bbox[:, 1]
        show_bbox_wh(cat_bbox_w, cat_bbox_h, classes, idx, out)

        # Figure out the ratio and area
        cat_ratio = cat_bbox_w / cat_bbox_h
        cat_area = cat_bbox_w * cat_bbox_h
        cat_ratio_nums = np.zeros((3, 1), dtype=int)
        cat_area_nums = np.zeros((3, 1), dtype=int)

        # Classification according to proportion rules
        for i in range(len(cat_ratio)):
            if cat_ratio[i] < 1:
                cat_ratio_nums[0] = cat_ratio_nums[0] + 1
            elif cat_ratio[i] == 1:
                cat_ratio_nums[1] = cat_ratio_nums[1] + 1
            elif cat_ratio[i] > 1:
                cat_ratio_nums[2] = cat_ratio_nums[2] + 1
        show_bbox_wh_ratio(cat_ratio_nums, classes, idx, out)

        # Classification according to area rules
        for t in range(len(cat_area)):
            if 0 <= cat_area[t] < 32**2:
                cat_area_nums[0] = cat_area_nums[0] + 1
            elif 32**2 <= cat_area[t] < 96**2:
                cat_area_nums[1] = cat_area_nums[1] + 1
            elif 96**2 <= cat_area[t] < 1e5**2:
                cat_area_nums[2] = cat_area_nums[2] + 1
        cat_area_num.append(list(np.concatenate(cat_area_nums)))
        progress_bar_classes.update()
    show_bbox_area(cfg, classes, cat_area_num, out)
    print('\nDraw End')


if __name__ == '__main__':
    main()
