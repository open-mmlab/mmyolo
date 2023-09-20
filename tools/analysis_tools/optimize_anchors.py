# Copyright (c) OpenMMLab. All rights reserved.
"""Optimize anchor settings on a specific dataset.

This script provides three methods to optimize YOLO anchors including k-means
anchor cluster, differential evolution and v5-k-means. You can use
``--algorithm k-means``, ``--algorithm differential_evolution`` and
``--algorithm v5-k-means`` to switch those methods.

Example:
    Use k-means anchor cluster::

        python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
        --algorithm k-means --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
        --out-dir ${OUT_DIR}

    Use differential evolution to optimize anchors::

        python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
        --algorithm differential_evolution \
        --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
        --out-dir ${OUT_DIR}

    Use v5-k-means to optimize anchors::

        python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
        --algorithm v5-k-means \
        --input-shape ${INPUT_SHAPE [WIDTH HEIGHT]} \
        --prior_match_thr ${PRIOR_MATCH_THR} \
        --out-dir ${OUT_DIR}
"""
import argparse

from mmdet.utils import replace_cfg_vals, update_data_root
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope

from mmyolo.registry import DATASETS
from mmyolo.utils import (YOLODEAnchorOptimizer, YOLOKMeansAnchorOptimizer,
                          YOLOV5KMeansAnchorOptimizer)


def parse_args():
    parser = argparse.ArgumentParser(description='Optimize anchor parameters.')
    parser.add_argument('config', help='Train config file path.')
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs='+',
        default=[640, 640],
        help='input image size, represent [width, height]')
    parser.add_argument(
        '--algorithm',
        default='DE',
        help='Algorithm used for anchor optimizing.'
        'Support k-means and differential_evolution for YOLO,'
        'and v5-k-means is special for YOLOV5.')
    parser.add_argument(
        '--iters',
        default=1000,
        type=int,
        help='Maximum iterations for optimizer.')
    parser.add_argument(
        '--prior-match-thr',
        default=4.0,
        type=float,
        help='anchor-label `gt_filter_sizes` ratio threshold '
        'hyperparameter used for training, default=4.0, this '
        'parameter is unique to v5-k-means')
    parser.add_argument(
        '--mutation-args',
        type=float,
        nargs='+',
        default=[0.9, 0.1],
        help='paramter of anchor optimize method genetic algorithm, '
        'represent [prob, sigma], this parameter is unique to v5-k-means')
    parser.add_argument(
        '--augment-args',
        type=float,
        nargs='+',
        default=[0.9, 1.1],
        help='scale factor of box size augment when metric box and anchor, '
        'represent [min, max], this parameter is unique to v5-k-means')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for calculating.')
    parser.add_argument(
        '--out-dir',
        default=None,
        type=str,
        help='Path to save anchor optimize result.')

    args = parser.parse_args()
    return args


def main():
    logger = MMLogger.get_current_instance()
    args = parse_args()
    cfg = args.config
    cfg = Config.fromfile(cfg)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    init_default_scope(cfg.get('default_scope', 'mmyolo'))

    input_shape = args.input_shape
    assert len(input_shape) == 2

    anchor_type = cfg.model.bbox_head.prior_generator.type
    assert anchor_type == 'mmdet.YOLOAnchorGenerator', \
        f'Only support optimize YOLOAnchor, but get {anchor_type}.'

    base_sizes = cfg.model.bbox_head.prior_generator.base_sizes
    num_anchor_per_level = [len(sizes) for sizes in base_sizes]

    train_data_cfg = cfg.train_dataloader
    while 'dataset' in train_data_cfg:
        train_data_cfg = train_data_cfg['dataset']
    dataset = DATASETS.build(train_data_cfg)

    if args.algorithm == 'k-means':
        optimizer = YOLOKMeansAnchorOptimizer(
            dataset=dataset,
            input_shape=input_shape,
            device=args.device,
            num_anchor_per_level=num_anchor_per_level,
            iters=args.iters,
            logger=logger,
            out_dir=args.out_dir)
    elif args.algorithm == 'DE':
        optimizer = YOLODEAnchorOptimizer(
            dataset=dataset,
            input_shape=input_shape,
            device=args.device,
            num_anchor_per_level=num_anchor_per_level,
            iters=args.iters,
            logger=logger,
            out_dir=args.out_dir)
    elif args.algorithm == 'v5-k-means':
        optimizer = YOLOV5KMeansAnchorOptimizer(
            dataset=dataset,
            input_shape=input_shape,
            device=args.device,
            num_anchor_per_level=num_anchor_per_level,
            iters=args.iters,
            prior_match_thr=args.prior_match_thr,
            mutation_args=args.mutation_args,
            augment_args=args.augment_args,
            logger=logger,
            out_dir=args.out_dir)
    else:
        raise NotImplementedError(
            f'Only support k-means and differential_evolution, '
            f'but get {args.algorithm}')

    optimizer.optimize()


if __name__ == '__main__':
    main()
