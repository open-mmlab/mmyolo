# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path

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
        '--func',
        default=1,
        type=int,
        help='func type: 1,2,3,4'
        'func 1:Distribution chart showing the number of '
        'categories and bbox instances'
        'func 2:Display the width and height distribution of '
        'each category and bbox instance respectively'
        'func 3:Display the width height scale distribution of '
        'each category and bbox instance separately'
        'func 4:Display the distribution map of '
        'category and bbox instance area')
    parser.add_argument(
        '--output-dir',
        default='./',
        type=str,
        help='If there is no display interface, you can save it'
        'Save pictures in./data/analysis by default')
    args = parser.parse_args()
    return args


def func_1(cfg, args, dataset):
    data_list = dataset.load_data_list()
    classes = dataset.metainfo['CLASSES']
    progress_bar = ProgressBar(len(dataset))

    cat_nums = np.zeros((len(classes), ), dtype=int)
    for item in data_list:
        for instance in item['instances']:
            cat_nums[
                instance['bbox_label']] = cat_nums[instance['bbox_label']] + 1
        progress_bar.update()

    fig = plt.figure(figsize=(50, 18), dpi=600)
    plt.bar(classes, cat_nums, align='center')
    plt.xticks(rotation=70)
    plt.xlabel('Category Name')
    plt.ylabel('Number of categories')
    plt.title(cfg.dataset_type)
    for x, y in enumerate(cat_nums):
        plt.text(x, y + 10, '%s' % y, ha='center', fontsize=4)
    out = args.output_dir
    if not os.path.exists(out):
        os.makedirs(out)
    fig.set_size_inches(35, 18)
    fig.savefig(f'{out}/{cfg.dataset_type}.jpg')  # Save Image


def func_2(cfg, args, dataset):
    data_list = dataset.load_data_list()
    classes = dataset.metainfo['CLASSES']
    progress_bar = ProgressBar(len(dataset))

    cat_with_bbox = [[] for _ in classes]
    for item in data_list:
        for instance in item['instances']:
            cat_with_bbox[instance['bbox_label']].append(instance['bbox'])
        progress_bar.update()
    for idx in range(len(classes)):
        which_cat_with_bbox = np.array(cat_with_bbox[idx])
        which_cat_w = which_cat_with_bbox[:, 2] - which_cat_with_bbox[:, 0]
        which_cat_h = which_cat_with_bbox[:, 3] - which_cat_with_bbox[:, 1]

        fig = plt.figure(figsize=(30, 18), dpi=80)
        plt.scatter(which_cat_w, which_cat_h, s=10)
        plt.xlabel('Width of bbox')
        plt.ylabel('High of bbox')
        plt.title(f'Current Display Category:{classes[idx]}')
        out = args.output_dir
        if not os.path.exists(out):
            os.makedirs(out)
        fig.savefig(f'{out}/{classes[idx]}.jpg')  # Save Image


def func_3(cfg, args, dataset):
    data_list = dataset.load_data_list()
    classes = dataset.metainfo['CLASSES']
    progress_bar = ProgressBar(len(dataset))

    cat_with_bbox = [[] for _ in classes]
    for item in data_list:
        for instance in item['instances']:
            cat_with_bbox[instance['bbox_label']].append(instance['bbox'])
        progress_bar.update()
    for idx in range(len(classes)):
        which_cat_with_bbox = np.array(cat_with_bbox[idx])
        which_cat_w = which_cat_with_bbox[:, 2] - which_cat_with_bbox[:, 0]
        which_cat_h = which_cat_with_bbox[:, 3] - which_cat_with_bbox[:, 1]
        which_cat_ratio = which_cat_w / which_cat_h
        less_equal = []
        equal = []
        greater_equal = []
        for i in range(len(which_cat_ratio)):
            ratios = round(which_cat_ratio[i], 2)
            if ratios < 1:
                less_equal.append(ratios)
            elif ratios == 1:
                equal.append(ratios)
            else:
                greater_equal.append(ratios)

        fig = plt.figure(figsize=(20, 18), dpi=100)
        labels = 'Less than equal', 'Equal', 'More than equal'
        values = len(less_equal), len(equal), len(greater_equal)
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
        out = args.output_dir
        if not os.path.exists(out):
            os.makedirs(out)
        fig.savefig(f'{out}/{classes[idx]}.jpg')  # Save Image


def func_4(cfg, args, dataset):
    data_list = dataset.load_data_list()
    classes = dataset.metainfo['CLASSES']
    progress_bar = ProgressBar(len(dataset))

    cat_with_bbox = [[] for _ in classes]
    cat_area_s_num = []
    cat_area_m_num = []
    cat_area_l_num = []
    for item in data_list:
        for instance in item['instances']:
            cat_with_bbox[instance['bbox_label']].append(instance['bbox'])
        progress_bar.update()
    for idx in range(len(classes)):
        which_cat_with_bbox = np.array(cat_with_bbox[idx])
        which_cat_w = which_cat_with_bbox[:, 2] - which_cat_with_bbox[:, 0]
        which_cat_h = which_cat_with_bbox[:, 3] - which_cat_with_bbox[:, 1]
        which_cat_area = which_cat_w * which_cat_h
        small = []
        medium = []
        large = []
        for i in range(len(which_cat_area)):
            area = which_cat_area[i]
            if area < 1024:
                small.append(area)
            elif 1024 < area < 9216:
                medium.append(area)
            else:
                large.append(area)
        cat_area_s_num.append(len(small))
        cat_area_m_num.append(len(medium))
        cat_area_l_num.append(len(large))

    x = np.arange(len(classes))
    width = 0.3
    fig = plt.figure(figsize=(50, 18), dpi=600)
    plt.bar(x - width, cat_area_s_num, width, label='Small', color='#438675')
    plt.bar(x, cat_area_m_num, width, label='Medium', color='#F7B469')
    plt.bar(x + width, cat_area_l_num, width, label='Large', color='#6BA6DA')
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
    for x1, y1 in enumerate(cat_area_s_num):
        plt.text(x1 - width, y1 + 8, y1, ha='center', fontsize=4)
    for x2, y2 in enumerate(cat_area_m_num):
        plt.text(x2, y2 + 5, y2, ha='center', fontsize=4)
    for x3, y3 in enumerate(cat_area_l_num):
        plt.text(x3 + width, y3 + 8, y3, ha='center', fontsize=4)
    out = args.output_dir
    if not os.path.exists(out):
        os.makedirs(out)
    fig.savefig(f'{out}/{cfg.dataset_type}.jpg')  # Save Image


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # register all modules in mmdet into the registries
    register_all_modules()

    # Build Dataset
    dataset = DATASETS.build(cfg.train_dataloader.dataset)

    # Import datasets according to different functions
    if args.func == 1:
        func_1(cfg, args, dataset)
    elif args.func == 2:
        func_2(cfg, args, dataset)
    elif args.func == 3:
        func_3(cfg, args, dataset)
    elif args.func == 4:
        func_4(cfg, args, dataset)


if __name__ == '__main__':
    main()
