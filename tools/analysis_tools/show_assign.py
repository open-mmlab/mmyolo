# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys

import mmcv
import numpy as np
from mmengine import ProgressBar
from mmengine.config import Config, DictAction
from mmengine.dataset import COLLATE_FUNCTIONS
from numpy import random

from mmyolo.registry import DATASETS, MODELS
from mmyolo.utils import register_all_modules
from mmyolo.visualization import DetAssignerVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMYOLO show the positive sample assign'
        ' results.')
    parser.add_argument(
        'config',
        default='configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py',
        help='config file path')
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
    parser.add_argument(
        '--show-number',
        '-n',
        type=int,
        default=sys.maxsize,
        help='number of images selected to visualize, '
        'must bigger than 0. if the number is bigger than length '
        'of dataset, show all the images in dataset; '
        'default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--output-dir',
        default='show',
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument('--show-prior', default=False, action='store_true')
    parser.add_argument('--not-show-label', default=False, action='store_true')
    parser.add_argument('--seed', default=-1, type=int, help='seed')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    register_all_modules()

    assert 'yolov5' in args.config, 'Now, this script only support yolov5.'

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
    model.eval()
    dataset_cfg = cfg.get('train_dataloader').get('dataset')
    dataset = DATASETS.build(dataset_cfg)

    # get collate_fn
    collate_fn_cfg = cfg.get('train_dataloader').pop(
        'collate_fn', dict(type='pseudo_collate'))
    collate_fn_type = collate_fn_cfg.pop('type')
    collate_fn = COLLATE_FUNCTIONS.get(collate_fn_type)

    # init visualizer
    visualizer = DetAssignerVisualizer(
        vis_backends=[{
            'type': 'LocalVisBackend'
        }], name='visualizer')
    visualizer.dataset_meta = dataset.metainfo
    # need priors size to draw priors
    visualizer.priors_size = model.bbox_head.prior_generator.base_anchors

    # make output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # init visualization image number
    assert args.show_number > 0
    display_number = min(args.show_number, len(dataset))

    progress_bar = ProgressBar(display_number)
    for ind_img in range(display_number):
        data = dataset.prepare_data(ind_img)

        # convert data to batch format
        batch_data = collate_fn([data])
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
