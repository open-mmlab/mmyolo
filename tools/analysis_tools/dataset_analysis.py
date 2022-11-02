# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path
from statistics import median

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mmengine.config import Config
from mmengine.utils import ProgressBar
from prettytable import PrettyTable

from mmyolo.registry import DATASETS
from mmyolo.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Distribution of categories and bbox instances')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--type',
        default='train',
        type=str,
        help='Dataset set type, e.g., "train" or "val"')
    parser.add_argument(
        '--class-name',
        default=None,
        type=str,
        help='Display category-specific data, e.g., "bicycle"')
    parser.add_argument(
        '--area-rule',
        default=[32, 96],
        type=int,
        nargs='+',
        help='Add new value and redefine regional rules,'
        ' e.g., 32 96 120,120 is new value')
    parser.add_argument(
        '--func',
        default=None,
        type=str,
        help='Dataset analysis function selection,'
        ' e.g., show_bbox_num')
    parser.add_argument(
        '--output-dir',
        default='./',
        type=str,
        help='Save address of dataset analysis visualization results')
    args = parser.parse_args()
    return args


def show_bbox_num(cfg, args, fig_set, class_name, class_num):
    """Display the distribution map of categories and number of bbox
    instances."""
    # Drawing function 1:
    # Draw designs
    fig = plt.figure(
        figsize=(fig_set['figsize'][0], fig_set['figsize'][1]), dpi=600)
    plt.bar(class_name, class_num, align='center')

    # Draw titles, labels and so on
    for x, y in enumerate(class_num):
        plt.text(x, y + 2, '%s' % y, ha='center', fontsize=fig_set['fontsize'])
    plt.xticks(rotation=fig_set['xticks_angle'])
    plt.xlabel('Category Name')
    plt.ylabel('Num of instances')
    plt.title(cfg.dataset_type)

    # Save figuer
    out_dir = os.path.join(args.output_dir, 'dataset_analysis')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = fig_set['out_name']
    fig.savefig(f'{out_dir}/{out_name}_bbox_num.jpg')  # Save Image
    plt.close()


def show_bbox_wh(args, fig_set, class_bbox_w, class_bbox_h, class_name):
    """Display the width and height distribution of categories and bbox
    instances."""
    # Drawing function 2:
    # Draw designs
    fig, ax = plt.subplots(
        figsize=(fig_set['figsize'][0], fig_set['figsize'][1]), dpi=600)

    # Set the position of the map and label on the x-axis
    positions_w = list(range(0, 12 * len(class_name), 12))
    positions_h = list(range(6, 12 * len(class_name), 12))
    positions_x_lable = list(range(3, 12 * len(class_name) + 1, 12))
    ax.violinplot(
        class_bbox_w, positions_w, showmeans=True, showmedians=True, widths=4)
    ax.violinplot(
        class_bbox_h, positions_h, showmeans=True, showmedians=True, widths=4)

    # Draw titles, labels and so on
    plt.xticks(rotation=fig_set['xticks_angle'])
    plt.ylabel('The width or height of bbox')
    plt.xlabel('Class name')
    plt.title('Width or height distribution of classes and bbox instances')

    # Draw the max, min and median of wide data in violin chart
    for i in range(len(class_bbox_w)):
        plt.text(
            positions_w[i],
            median(class_bbox_w[i]),
            f'{"%.2f" % median(class_bbox_w[i])}',
            ha='center',
            fontsize=fig_set['fontsize'])
        plt.text(
            positions_w[i],
            max(class_bbox_w[i]),
            f'{"%.2f" % max(class_bbox_w[i])}',
            ha='center',
            fontsize=fig_set['fontsize'])
        plt.text(
            positions_w[i],
            min(class_bbox_w[i]),
            f'{"%.2f" % min(class_bbox_w[i])}',
            ha='center',
            fontsize=fig_set['fontsize'])

    # Draw the max, min and median of height data in violin chart
    for i in range(len(positions_h)):
        plt.text(
            positions_h[i],
            median(class_bbox_h[i]),
            f'{"%.2f" % median(class_bbox_h[i])}',
            ha='center',
            fontsize=fig_set['fontsize'])
        plt.text(
            positions_h[i],
            max(class_bbox_h[i]),
            f'{"%.2f" % max(class_bbox_h[i])}',
            ha='center',
            fontsize=fig_set['fontsize'])
        plt.text(
            positions_h[i],
            min(class_bbox_h[i]),
            f'{"%.2f" % min(class_bbox_h[i])}',
            ha='center',
            fontsize=fig_set['fontsize'])

    # Draw Legend
    plt.setp(ax, xticks=positions_x_lable, xticklabels=class_name)
    labels = ['bbox_w', 'bbox_h']
    colors = ['steelblue', 'darkorange']
    patches = [
        mpatches.Patch(color=colors[i], label=f'{labels[i]:s}')
        for i in range(len(colors))
    ]
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='upper center', handles=patches, ncol=2)

    # Save figuer
    out_dir = os.path.join(args.output_dir, 'dataset_analysis')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = fig_set['out_name']
    fig.savefig(f'{out_dir}/{out_name}_bbox_wh.jpg')  # Save Image
    plt.close()


def show_bbox_wh_ratio(args, fig_set, class_name, class_bbox_ratio):
    """Display the distribution map of category and bbox instance width and
    height ratio."""
    # Drawing function 3:
    # Draw designs
    fig, ax = plt.subplots(
        figsize=(fig_set['figsize'][0], fig_set['figsize'][1]), dpi=600)

    # Set the position of the map and label on the x-axis
    positions = list(range(0, 6 * len(class_name), 6))
    ax.violinplot(
        class_bbox_ratio,
        positions,
        showmeans=True,
        showmedians=True,
        widths=5)

    # Draw titles, labels and so on
    plt.xticks(rotation=fig_set['xticks_angle'])
    plt.ylabel('Ratio of width to height of bbox')
    plt.xlabel('Class name')
    plt.title('Width to height ratio distribution of class and bbox instances')

    # Draw the max, min and median of wide data in violin chart
    for i in range(len(class_bbox_ratio)):
        plt.text(
            positions[i],
            median(class_bbox_ratio[i]),
            f'{"%.2f" % median(class_bbox_ratio[i])}',
            ha='center',
            fontsize=fig_set['fontsize'])
        plt.text(
            positions[i],
            max(class_bbox_ratio[i]),
            f'{"%.2f" % max(class_bbox_ratio[i])}',
            ha='center',
            fontsize=fig_set['fontsize'])
        plt.text(
            positions[i],
            min(class_bbox_ratio[i]),
            f'{"%.2f" % min(class_bbox_ratio[i])}',
            ha='center',
            fontsize=fig_set['fontsize'])

    # Set the position of the map and label on the x-axis
    plt.setp(ax, xticks=positions, xticklabels=class_name)

    # Save figuer
    out_dir = os.path.join(args.output_dir, 'dataset_analysis')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = fig_set['out_name']
    fig.savefig(f'{out_dir}/{out_name}_bbox_ratio.jpg')  # Save Image
    plt.close()


def show_bbox_area(args, fig_set, area_rule, class_name, bbox_area_num):
    """Display the distribution map of category and bbox instance area based on
    the rules of large, medium and small objects."""
    # Drawing function 4:
    # Set the direct distance of each label and the width of each histogram
    # Set the required labels and colors
    positions = np.arange(0, 2 * len(class_name), 2)
    width = 0.4
    labels = ['Small', 'Mediun', 'Large', 'Huge']
    colors = ['#438675', '#F7B469', '#6BA6DA', '#913221']

    # Draw designs
    fig = plt.figure(
        figsize=(fig_set['figsize'][0], fig_set['figsize'][1]), dpi=600)
    for i in range(len(area_rule) - 1):
        area_num = [bbox_area_num[idx][i] for idx in range(len(class_name))]
        plt.bar(
            positions + width * i,
            area_num,
            width,
            label=labels[i],
            color=colors[i])
        for idx, (x, y) in enumerate(zip(positions.tolist(), area_num)):
            plt.text(
                x + width * i, y, y, ha='center', fontsize=fig_set['fontsize'])

    # Draw titles, labels and so on
    plt.xticks(rotation=fig_set['xticks_angle'])
    plt.xticks(positions + width * ((len(area_rule) - 2) / 2), class_name)
    plt.ylabel('Class Area')
    plt.xlabel('Class Name')
    plt.title(
        'Area and number of large, medium and small objects of each class')

    # Set and Draw Legend
    patches = [
        mpatches.Patch(color=colors[i], label=f'{labels[i]:s}')
        for i in range(len(area_rule) - 1)
    ]
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    ax.legend(loc='upper center', handles=patches, ncol=len(area_rule) - 1)

    # Save figuer
    out_dir = os.path.join(args.output_dir, 'dataset_analysis')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = fig_set['out_name']
    fig.savefig(f'{out_dir}/{out_name}_bbox_area.jpg')  # Save Image
    plt.close()


def show_class_list(class_name, class_num):
    """Print the name of the dataset class and its corresponding quantity."""

    data_info = PrettyTable()
    data_info.title = 'Dataset analysis'
    # List Print Settings
    # If the quantity is too large, 25 rows will be displayed in each column
    if len(class_name) < 25:
        data_info.add_column('Class name', class_name)
        data_info.add_column('bbox_num', class_num)
    elif len(class_name) % 25 != 0 and len(class_name) > 25:
        col_num = int(len(class_name) / 25) + 1
        class_nums = class_num.tolist()
        for i in range(0, (col_num * 25) - len(class_name)):
            class_name.append('')
            class_nums.append('')
        for i in range(0, len(class_name), 25):
            data_info.add_column('Class name', class_name[i:i + 25])
            data_info.add_column('Bbox num', class_nums[i:i + 25])

    # Align display data to the left
    data_info.align['Class name'] = 'l'
    data_info.align['Bbox num'] = 'l'
    print(data_info)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # register all modules in mmdet into the registries
    register_all_modules()

    # 1.Build Dataset
    if args.type == 'train':
        dataset = DATASETS.build(cfg.train_dataloader.dataset)
    elif args.type == 'val':
        dataset = DATASETS.build(cfg.val_dataloader.dataset)
    else:
        raise RuntimeError(
            'Please enter the correct data type, e.g., train or val')
    data_list = dataset.load_data_list()

    # 2.Prepare data
    progress_bar = ProgressBar(len(dataset))

    # Drawing settings
    fig_all_set = {
        'figsize': [45, 18],
        'fontsize': 4,
        'xticks_angle': 70,
        'out_name': cfg.dataset_type
    }
    fig_one_set = {
        'figsize': [15, 10],
        'fontsize': 10,
        'xticks_angle': 0,
        'out_name': args.class_name
    }

    # Call the category name and save address
    if args.class_name is None:
        classes = dataset.metainfo['CLASSES']
        classes_idx = [i for i in range(len(classes))]
        fig_set = fig_all_set
    elif args.class_name in dataset.metainfo['CLASSES']:
        classes = [args.class_name]
        classes_idx = [dataset.metainfo['CLASSES'].index(args.class_name)]
        fig_set = fig_one_set
    else:
        raise RuntimeError('Please enter the correct class name, e.g., person')

    # Building Area Rules
    if 32 in args.area_rule and 96 in args.area_rule and len(
            args.area_rule) <= 3:
        area_rules = [0] + args.area_rule + [1e5]
    else:
        raise RuntimeError(
            'Please enter the correct area rule, e.g., 32 96 120')
    area_rule = sorted(area_rules)

    # Build arrays or lists to store data for each category
    class_num = np.zeros((len(classes), ), dtype=np.int64)
    class_bbox = [[] for _ in classes]
    class_name = []
    class_bbox_w = []
    class_bbox_h = []
    class_bbox_ratio = []
    bbox_area_num = []

    # Get the quantity and bbox data corresponding to each category
    for img in data_list:
        for instance in img['instances']:
            if instance[
                    'bbox_label'] in classes_idx and args.class_name is None:
                class_num[instance['bbox_label']] += 1
                class_bbox[instance['bbox_label']].append(instance['bbox'])
            elif instance['bbox_label'] in classes_idx and args.class_name:
                class_num[0] += 1
                class_bbox[0].append(instance['bbox'])
        progress_bar.update()

    # Get the width, height and area of bbox corresponding to each category
    print('\n\nStart drawing')
    progress_bar_classes = ProgressBar(len(classes))
    for idx, (classes, classes_idx) in enumerate(zip(classes, classes_idx)):
        bbox = np.array(class_bbox[idx])
        bbox_wh = bbox[:, 2:4] - bbox[:, 0:2]
        bbox_ratio = bbox_wh[:, 0] / bbox_wh[:, 1]
        bbox_area = bbox_wh[:, 0] * bbox_wh[:, 1]

        class_name.append(classes)
        class_bbox_w.append(bbox_wh[:, 0].tolist())
        class_bbox_h.append(bbox_wh[:, 1].tolist())
        class_bbox_ratio.append(bbox_ratio)

        bbox_area_nums = np.zeros((len(area_rule) - 1, ), dtype=np.int64)
        for i in range(len(area_rule) - 1):
            bbox_area_nums[i] = np.logical_and(
                bbox_area >= area_rule[i]**2,
                bbox_area < area_rule[i + 1]**2).sum()
        bbox_area_num.append(bbox_area_nums.tolist())
        progress_bar_classes.update()

    # 3.draw Dataset Information
    if args.func is None:
        show_bbox_num(cfg, args, fig_set, class_name, class_num)
        show_bbox_wh(args, fig_set, class_bbox_w, class_bbox_h, class_name)
        show_bbox_wh_ratio(args, fig_set, class_name, class_bbox_ratio)
        show_bbox_area(args, fig_set, area_rule, class_name, bbox_area_num)
    elif args.func == 'show_bbox_num':
        show_bbox_num(cfg, args, fig_set, class_name, class_num)
    elif args.func == 'show_bbox_wh':
        show_bbox_wh(args, fig_set, class_bbox_w, class_bbox_h, class_name)
    elif args.func == 'show_bbox_wh_ratio':
        show_bbox_wh_ratio(args, fig_set, class_name, class_bbox_ratio)
    elif args.func == 'show_bbox_area':
        show_bbox_area(args, fig_set, area_rule, class_name, bbox_area_num)
    else:
        raise RuntimeError(
            'Please enter the correct func name, e.g., show_bbox_num')

    print('\nDraw End\n')

    # 4.Print Dataset Information
    show_class_list(class_name, class_num)


if __name__ == '__main__':
    main()
