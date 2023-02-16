# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys
import warnings

import mmcv
import numpy as np
import torch
from mmengine import ProgressBar
from mmengine.config import Config, DictAction
from mmengine.dataset import COLLATE_FUNCTIONS
from mmengine.runner.checkpoint import load_checkpoint
from numpy import random

from mmyolo.registry import DATASETS, MODELS
from mmyolo.utils import register_all_modules
from projects.assigner_visualization.dense_heads import (RTMHeadAssigner,
                                                         YOLOv5HeadAssigner,
                                                         YOLOv7HeadAssigner,
                                                         YOLOv8HeadAssigner)
from projects.assigner_visualization.visualization import \
    YOLOAssignerVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMYOLO show the positive sample assigning'
        ' results.')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', '-c', type=str, help='checkpoint file')
    parser.add_argument(
        '--show-number',
        '-n',
        type=int,
        default=sys.maxsize,
        help='number of images selected to save, '
        'must bigger than 0. if the number is bigger than length '
        'of dataset, show all the images in dataset; '
        'default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--output-dir',
        default='assigned_results',
        type=str,
        help='The name of the folder where the image is saved.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--show-prior',
        default=False,
        action='store_true',
        help='Whether to show prior on image.')
    parser.add_argument(
        '--not-show-label',
        default=False,
        action='store_true',
        help='Whether to show label on image.')
    parser.add_argument('--seed', default=-1, type=int, help='random seed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    register_all_modules()

    # set random seed
    seed = int(args.seed)
    if seed != -1:
        print(f'Set the global seed: {seed}')
        random.seed(int(args.seed))

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build model
    model = MODELS.build(cfg.model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint)
    elif isinstance(model.bbox_head, (YOLOv7HeadAssigner, RTMHeadAssigner)):
        warnings.warn(
            'if you use dynamic_assignment methods such as YOLOv7 or '
            'YOLOv8 or RTMDet assigner, please load the checkpoint.')
    assert isinstance(model.bbox_head, (YOLOv5HeadAssigner,
                                        YOLOv7HeadAssigner,
                                        YOLOv8HeadAssigner,
                                        RTMHeadAssigner)), \
        'Now, this script only support YOLOv5, YOLOv7, YOLOv8 and RTMdet, ' \
        'and bbox_head must use ' \
        '`YOLOv5HeadAssigner or YOLOv7HeadAssigne or YOLOv8HeadAssigner ' \
        'or RTMHeadAssigner`. Please use `' \
        'yolov5_s-v61_syncbn_fast_8xb16-300e_coco_assignervisualization.py' \
        'or yolov7_tiny_syncbn_fast_8x16b-300e_coco_assignervisualization.py' \
        'or yolov8_s_syncbn_fast_8xb16-500e_coco_assignervisualization.py' \
        'or rtmdet_s_syncbn_fast_8xb32-300e_coco_assignervisualization.py' \
        """` as config file."""
    model.eval()
    model.to(args.device)

    # build dataset
    dataset_cfg = cfg.get('train_dataloader').get('dataset')
    dataset = DATASETS.build(dataset_cfg)

    # get collate_fn
    collate_fn_cfg = cfg.get('train_dataloader').pop(
        'collate_fn', dict(type='pseudo_collate'))
    collate_fn_type = collate_fn_cfg.pop('type')
    collate_fn = COLLATE_FUNCTIONS.get(collate_fn_type)

    # init visualizer
    visualizer = YOLOAssignerVisualizer(
        vis_backends=[{
            'type': 'LocalVisBackend'
        }], name='visualizer')
    visualizer.dataset_meta = dataset.metainfo
    # need priors size to draw priors

    if hasattr(model.bbox_head.prior_generator, 'base_anchors'):
        visualizer.priors_size = model.bbox_head.prior_generator.base_anchors

    # make output dir
    os.makedirs(args.output_dir, exist_ok=True)
    print('Results will save to ', args.output_dir)

    # init visualization image number
    assert args.show_number > 0
    display_number = min(args.show_number, len(dataset))

    progress_bar = ProgressBar(display_number)
    for ind_img in range(display_number):
        data = dataset.prepare_data(ind_img)
        if data is None:
            print('Unable to visualize {} due to strong data augmentations'.
                  format(dataset[ind_img]['data_samples'].img_path))
            continue
        # convert data to batch format
        batch_data = collate_fn([data])
        with torch.no_grad():
            assign_results = model.assign(batch_data)

        img = data['inputs'].cpu().numpy().astype(np.uint8).transpose(
            (1, 2, 0))
        # bgr2rgb
        img = mmcv.bgr2rgb(img)

        gt_instances = data['data_samples'].gt_instances

        img_show = visualizer.draw_assign(img, assign_results, gt_instances,
                                          args.show_prior, args.not_show_label)

        if hasattr(data['data_samples'], 'img_path'):
            filename = osp.basename(data['data_samples'].img_path)
        else:
            # some dataset have not image path
            filename = f'{ind_img}.jpg'
        out_file = osp.join(args.output_dir, filename)

        # convert rgb 2 bgr and save img
        mmcv.imwrite(mmcv.rgb2bgr(img_show), out_file)
        progress_bar.update()


if __name__ == '__main__':
    main()
