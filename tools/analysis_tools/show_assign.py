# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import random
import sys

import cv2
import numpy as np
import mmcv
from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.visualization import DetLocalVisualizer
from mmengine import ProgressBar
from mmengine.config import Config, DictAction
from mmengine.dataset import COLLATE_FUNCTIONS
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner
from matplotlib import pyplot as plt

from mmyolo.datasets import YOLOv5CocoDataset
from mmyolo.registry import RUNNERS, MODELS, DATASETS, VISUALIZERS
from mmyolo.utils import register_all_modules
from mmyolo.visualization import DetAssignerVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMYOLO test (and eval) a model')
    parser.add_argument('config', help='test config file path')
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
        default=None,
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument('--seed', default=-1, type=int, help='seed')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    register_all_modules()

    assert 'yolov5' in args.config, 'Now, this script only support yolov5.'

    seed = int(args.seed)
    if seed != -1:
        print(f'Set the global seed: {seed}')
        np.random.seed(int(args.seed))

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # build model
    model = MODELS.build(cfg.model)
    model.eval()
    dataset_cfg = cfg.get('train_dataloader').get('dataset')
    dataset = DATASETS.build(dataset_cfg)    # type: YOLOv5CocoDataset

    # get collate_fn
    collate_fn_cfg = cfg.get('train_dataloader').pop('collate_fn',
                                                     dict(type='pseudo_collate'))
    collate_fn_type = collate_fn_cfg.pop('type')
    collate_fn = COLLATE_FUNCTIONS.get(collate_fn_type)

    # init visualizer
    # cfg.merge_from_dict(dict(visualizer=dict(type='DetAssignerVisualizer')))
    # visualizer = VISUALIZERS.build(cfg['visualizer'])   # type: DetAssignerVisualizer
    # visualizer.dataset_meta = dataset.metainfo
    visualizer = DetAssignerVisualizer(vis_backends=[{'type': 'LocalVisBackend'}], name='visualizer')
    visualizer.dataset_meta = dataset.metainfo
    visualizer.priors_size = model.bbox_head.prior_generator.base_anchors
    # visualizer = DetLocalVisualizer(vis_backends=[{'type': 'LocalVisBackend'}], name='visualizer')

    # init visualization image number
    assert args.show_number > 0
    display_number = min(args.show_number, len(dataset))

    progress_bar = ProgressBar(display_number)
    for ind_img in range(display_number):
        # data_info = dataset.get_data_info(idx)
        data = dataset.prepare_data(ind_img)

        batch_data = collate_fn([data])
        assign_results = model.assign(batch_data)

        img = data['inputs'].cpu().numpy().astype(np.uint8).transpose((1, 2, 0))
        img = mmcv.bgr2rgb(img)

        gt_instances = data['data_samples'].gt_instances

        # gt_labels = data['data_samples'].gt_instances.labels
        # gt_bboxes = data['data_samples'].gt_instances.bboxes

        for ind_feat, assign_results_feat in enumerate(assign_results):
            for ind_prior, assign_results_prior in enumerate(assign_results_feat):
                img_show = visualizer.draw_assign(img, assign_results_prior, gt_instances, ind_feat)

                if hasattr(data['data_samples'], 'img_path'):
                    filename = osp.basename(data['data_samples'].img_path)
                    filename = f'{osp.splitext(filename)[0]}_feat{ind_feat}_prior{ind_prior}{osp.splitext(filename)[1]}'
                else:
                    # some dataset have not image path
                    filename = f'{ind_img}_feat{ind_feat}_prior{ind_prior}.jpg'
                out_file = osp.join(args.output_dir,
                                    filename) if args.output_dir is not None else None

                if out_file is not None:
                    mmcv.imwrite(img_show[..., ::-1], out_file)

                if not args.not_show:
                    visualizer.show(
                        img_show, win_name=filename, wait_time=args.show_interval)

                progress_bar.update()




if __name__ == '__main__':
    main()