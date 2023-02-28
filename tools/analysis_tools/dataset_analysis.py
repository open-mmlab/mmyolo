# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path
from statistics import median

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar
from prettytable import PrettyTable

from mmyolo.registry import DATASETS
from mmyolo.utils.misc import show_data_classes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Distribution of categories and bbox instances')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--val-dataset',
        default=False,
        action='store_true',
        help='The default train_dataset.'
        'To change it to val_dataset, enter "--val-dataset"')
    parser.add_argument(
        '--class-name',
        default=None,
        type=str,
        help='Display specific class, e.g., "bicycle"')
    parser.add_argument(
        '--area-rule',
        default=None,
        type=int,
        nargs='+',
        help='Redefine area rules,but no more than three numbers.'
        ' e.g., 30 70 125')
    parser.add_argument(
        '--func',
        default=None,
        type=str,
        choices=[
            'show_bbox_num', 'show_bbox_wh', 'show_bbox_wh_ratio',
            'show_bbox_area'
        ],
        help='Dataset analysis function selection.')
    parser.add_argument(
        '--out-dir',
        default='./dataset_analysis',
        type=str,
        help='Output directory of dataset analysis visualization results,'
        ' Save in "./dataset_analysis/" by default')
    args = parser.parse_args()
    return args


def show_bbox_num(cfg, out_dir, fig_set, class_name, class_num):
    """Display the distribution map of categories and number of bbox
    instances."""
    print('\n\nDrawing bbox_num figure:')
    # Draw designs
    fig = plt.figure(
        figsize=(fig_set['figsize'][0], fig_set['figsize'][1]), dpi=300)
    plt.bar(class_name, class_num, align='center')

    # Draw titles, labels and so on
    for x, y in enumerate(class_num):
        plt.text(x, y, '%s' % y, ha='center', fontsize=fig_set['fontsize'] + 3)
    plt.xticks(rotation=fig_set['xticks_angle'])
    plt.xlabel('Category Name')
    plt.ylabel('Num of instances')
    plt.title(cfg.dataset_type)

    # Save figure
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = fig_set['out_name']
    fig.savefig(
        f'{out_dir}/{out_name}_bbox_num.jpg',
        bbox_inches='tight',
        pad_inches=0.1)  # Save Image
    plt.close()
    print(f'End and save in {out_dir}/{out_name}_bbox_num.jpg')


def show_bbox_wh(out_dir, fig_set, class_bbox_w, class_bbox_h, class_name):
    """Display the width and height distribution of categories and bbox
    instances."""
    print('\n\nDrawing bbox_wh figure:')
    # Draw designs
    fig, ax = plt.subplots(
        figsize=(fig_set['figsize'][0], fig_set['figsize'][1]), dpi=300)

    # Set the position of the map and label on the x-axis
    positions_w = list(range(0, 12 * len(class_name), 12))
    positions_h = list(range(6, 12 * len(class_name), 12))
    positions_x_label = list(range(3, 12 * len(class_name) + 1, 12))
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
    plt.setp(ax, xticks=positions_x_label, xticklabels=class_name)
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

    # Save figure
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = fig_set['out_name']
    fig.savefig(
        f'{out_dir}/{out_name}_bbox_wh.jpg',
        bbox_inches='tight',
        pad_inches=0.1)  # Save Image
    plt.close()
    print(f'End and save in {out_dir}/{out_name}_bbox_wh.jpg')


def show_bbox_wh_ratio(out_dir, fig_set, class_name, class_bbox_ratio):
    """Display the distribution map of category and bbox instance width and
    height ratio."""
    print('\n\nDrawing bbox_wh_ratio figure:')
    # Draw designs
    fig, ax = plt.subplots(
        figsize=(fig_set['figsize'][0], fig_set['figsize'][1]), dpi=300)

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

    # Save figure
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = fig_set['out_name']
    fig.savefig(
        f'{out_dir}/{out_name}_bbox_ratio.jpg',
        bbox_inches='tight',
        pad_inches=0.1)  # Save Image
    plt.close()
    print(f'End and save in {out_dir}/{out_name}_bbox_ratio.jpg')


def show_bbox_area(out_dir, fig_set, area_rule, class_name, bbox_area_num):
    """Display the distribution map of category and bbox instance area based on
    the rules of large, medium and small objects."""
    print('\n\nDrawing bbox_area figure:')
    # Set the direct distance of each label and the width of each histogram
    # Set the required labels and colors
    positions = np.arange(0, 2 * len(class_name), 2)
    width = 0.4
    labels = ['Small', 'Mediun', 'Large', 'Huge']
    colors = ['#438675', '#F7B469', '#6BA6DA', '#913221']

    # Draw designs
    fig = plt.figure(
        figsize=(fig_set['figsize'][0], fig_set['figsize'][1]), dpi=300)
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
                x + width * i,
                y,
                y,
                ha='center',
                fontsize=fig_set['fontsize'] - 1)

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

    # Save figure
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = fig_set['out_name']
    fig.savefig(
        f'{out_dir}/{out_name}_bbox_area.jpg',
        bbox_inches='tight',
        pad_inches=0.1)  # Save Image
    plt.close()
    print(f'End and save in {out_dir}/{out_name}_bbox_area.jpg')


def show_class_list(classes, class_num):
    """Print the data of the class obtained by the current run."""
    print('\n\nThe information obtained is as follows:')
    class_info = PrettyTable()
    class_info.title = 'Information of dataset class'
    # List Print Settings
    # If the quantity is too large, 25 rows will be displayed in each column
    if len(classes) < 25:
        class_info.add_column('Class name', classes)
        class_info.add_column('Bbox num', class_num)
    elif len(classes) % 25 != 0 and len(classes) > 25:
        col_num = int(len(classes) / 25) + 1
        class_nums = class_num.tolist()
        class_name_list = list(classes)
        for i in range(0, (col_num * 25) - len(classes)):
            class_name_list.append('')
            class_nums.append('')
        for i in range(0, len(class_name_list), 25):
            class_info.add_column('Class name', class_name_list[i:i + 25])
            class_info.add_column('Bbox num', class_nums[i:i + 25])

    # Align display data to the left
    class_info.align['Class name'] = 'l'
    class_info.align['Bbox num'] = 'l'
    print(class_info)


def show_data_list(args, area_rule):
    """Print run setup information."""
    print('\n\nPrint current running information:')
    data_info = PrettyTable()
    data_info.title = 'Dataset information'
    # Print the corresponding information according to the settings
    if args.val_dataset is False:
        data_info.add_column('Dataset type', ['train_dataset'])
    elif args.val_dataset is True:
        data_info.add_column('Dataset type', ['val_dataset'])
    if args.class_name is None:
        data_info.add_column('Class name', ['All classes'])
    else:
        data_info.add_column('Class name', [args.class_name])
    if args.func is None:
        data_info.add_column('Function', ['All function'])
    else:
        data_info.add_column('Function', [args.func])
    data_info.add_column('Area rule', [area_rule])

    print(data_info)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    init_default_scope(cfg.get('default_scope', 'mmyolo'))

    def replace_pipeline_to_none(cfg):
        """Recursively iterate over all dataset(or datasets) and set their
        pipelines to none.Datasets are mean ConcatDataset.

        Recursively terminates only when all dataset(or datasets) have been
        traversed
        """

        if cfg.get('dataset', None) is None and cfg.get('datasets',
                                                        None) is None:
            return
        dataset = cfg.dataset if cfg.get('dataset', None) else cfg.datasets
        if isinstance(dataset, list):
            for item in dataset:
                item.pipeline = None
        elif dataset.get('pipeline', None):
            dataset.pipeline = None
        else:
            replace_pipeline_to_none(dataset)

    # 1.Build Dataset
    if args.val_dataset is False:
        replace_pipeline_to_none(cfg.train_dataloader)
        dataset = DATASETS.build(cfg.train_dataloader.dataset)
    else:
        replace_pipeline_to_none(cfg.val_dataloader)
        dataset = DATASETS.build(cfg.val_dataloader.dataset)

    # 2.Prepare data
    # Drawing settings
    fig_all_set = {
        'figsize': [35, 18],
        'fontsize': int(10 - 0.08 * len(dataset.metainfo['classes'])),
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
        classes = dataset.metainfo['classes']
        classes_idx = [i for i in range(len(classes))]
        fig_set = fig_all_set
    elif args.class_name in dataset.metainfo['classes']:
        classes = [args.class_name]
        classes_idx = [dataset.metainfo['classes'].index(args.class_name)]
        fig_set = fig_one_set
    else:
        data_classes = dataset.metainfo['classes']
        show_data_classes(data_classes)
        raise RuntimeError(f'Expected args.class_name to be one of the list,'
                           f'but got "{args.class_name}"')

    # Building Area Rules
    if args.area_rule is None:
        area_rule = [0, 32, 96, 1e5]
    elif args.area_rule and len(args.area_rule) <= 3:
        area_rules = [0] + args.area_rule + [1e5]
        area_rule = sorted(area_rules)
    else:
        raise RuntimeError(
            f'Expected the "{args.area_rule}" to be e.g. 30 60 120, '
            'and no more than three numbers.')

    # Build arrays or lists to store data for each category
    class_num = np.zeros((len(classes), ), dtype=np.int64)
    class_bbox = [[] for _ in classes]
    class_name = []
    class_bbox_w = []
    class_bbox_h = []
    class_bbox_ratio = []
    bbox_area_num = []

    show_data_list(args, area_rule)
    # Get the quantity and bbox data corresponding to each category
    print('\nRead the information of each picture in the dataset:')
    progress_bar = ProgressBar(len(dataset))
    for index in range(len(dataset)):
        for instance in dataset[index]['instances']:
            if instance[
                    'bbox_label'] in classes_idx and args.class_name is None:
                class_num[instance['bbox_label']] += 1
                class_bbox[instance['bbox_label']].append(instance['bbox'])
            elif instance['bbox_label'] in classes_idx and args.class_name:
                class_num[0] += 1
                class_bbox[0].append(instance['bbox'])
        progress_bar.update()
    show_class_list(classes, class_num)
    # Get the width, height and area of bbox corresponding to each category
    print('\nRead bbox information in each class:')
    progress_bar_classes = ProgressBar(len(classes))
    for idx, (classes, classes_idx) in enumerate(zip(classes, classes_idx)):
        bbox = np.array(class_bbox[idx])
        bbox_area_nums = np.zeros((len(area_rule) - 1, ), dtype=np.int64)
        if len(bbox) > 0:
            bbox_wh = bbox[:, 2:4] - bbox[:, 0:2]
            bbox_ratio = bbox_wh[:, 0] / bbox_wh[:, 1]
            bbox_area = bbox_wh[:, 0] * bbox_wh[:, 1]
            class_bbox_w.append(bbox_wh[:, 0].tolist())
            class_bbox_h.append(bbox_wh[:, 1].tolist())
            class_bbox_ratio.append(bbox_ratio.tolist())

            # The area rule, there is an section between two numbers
            for i in range(len(area_rule) - 1):
                bbox_area_nums[i] = np.logical_and(
                    bbox_area >= area_rule[i]**2,
                    bbox_area < area_rule[i + 1]**2).sum()
        elif len(bbox) == 0:
            class_bbox_w.append([0])
            class_bbox_h.append([0])
            class_bbox_ratio.append([0])

        class_name.append(classes)
        bbox_area_num.append(bbox_area_nums.tolist())
        progress_bar_classes.update()

    # 3.draw Dataset Information
    if args.func is None:
        show_bbox_num(cfg, args.out_dir, fig_set, class_name, class_num)
        show_bbox_wh(args.out_dir, fig_set, class_bbox_w, class_bbox_h,
                     class_name)
        show_bbox_wh_ratio(args.out_dir, fig_set, class_name, class_bbox_ratio)
        show_bbox_area(args.out_dir, fig_set, area_rule, class_name,
                       bbox_area_num)
    elif args.func == 'show_bbox_num':
        show_bbox_num(cfg, args.out_dir, fig_set, class_name, class_num)
    elif args.func == 'show_bbox_wh':
        show_bbox_wh(args.out_dir, fig_set, class_bbox_w, class_bbox_h,
                     class_name)
    elif args.func == 'show_bbox_wh_ratio':
        show_bbox_wh_ratio(args.out_dir, fig_set, class_name, class_bbox_ratio)
    elif args.func == 'show_bbox_area':
        show_bbox_area(args.out_dir, fig_set, area_rule, class_name,
                       bbox_area_num)
    else:
        raise RuntimeError(
            'Please enter the correct func name, e.g., show_bbox_num')


if __name__ == '__main__':
    main()
