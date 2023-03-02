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
import os.path as osp
import random
from typing import Tuple

import numpy as np
import torch
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import replace_cfg_vals, update_data_root
from mmengine.config import Config
from mmengine.fileio import dump
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar
from scipy.optimize import differential_evolution
from torch import Tensor

from mmyolo.registry import DATASETS

try:
    from scipy.cluster.vq import kmeans
except ImportError:
    kmeans = None


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


class BaseAnchorOptimizer:
    """Base class for anchor optimizer.

    Args:
        dataset (obj:`Dataset`): Dataset object.
        input_shape (list[int]): Input image shape of the model.
            Format in [width, height].
        num_anchor_per_level (list[int]) : Number of anchors for each level.
        logger (obj:`logging.Logger`): The logger for logging.
        device (str, optional): Device used for calculating.
            Default: 'cuda:0'
        out_dir (str, optional): Path to save anchor optimize result.
            Default: None
    """

    def __init__(self,
                 dataset,
                 input_shape,
                 num_anchor_per_level,
                 logger,
                 device='cuda:0',
                 out_dir=None):
        self.dataset = dataset
        self.input_shape = input_shape
        self.num_anchor_per_level = num_anchor_per_level
        self.num_anchors = sum(num_anchor_per_level)
        self.logger = logger
        self.device = device
        self.out_dir = out_dir
        bbox_whs, img_shapes = self.get_whs_and_shapes()
        ratios = img_shapes.max(1, keepdims=True) / np.array([input_shape])

        # resize to input shape
        self.bbox_whs = bbox_whs / ratios

    def get_whs_and_shapes(self):
        """Get widths and heights of bboxes and shapes of images.

        Returns:
            tuple[np.ndarray]: Array of bbox shapes and array of image
            shapes with shape (num_bboxes, 2) in [width, height] format.
        """
        self.logger.info('Collecting bboxes from annotation...')
        bbox_whs = []
        img_shapes = []
        prog_bar = ProgressBar(len(self.dataset))
        for idx in range(len(self.dataset)):
            data_info = self.dataset.get_data_info(idx)
            img_shape = np.array([data_info['width'], data_info['height']])
            gt_instances = data_info['instances']
            for instance in gt_instances:
                bbox = np.array(instance['bbox'])
                gt_filter_sizes = bbox[2:4] - bbox[0:2]
                img_shapes.append(img_shape)
                bbox_whs.append(gt_filter_sizes)

            prog_bar.update()
        print('\n')
        bbox_whs = np.array(bbox_whs)
        img_shapes = np.array(img_shapes)
        self.logger.info(f'Collected {bbox_whs.shape[0]} bboxes.')
        return bbox_whs, img_shapes

    def get_zero_center_bbox_tensor(self):
        """Get a tensor of bboxes centered at (0, 0).

        Returns:
            Tensor: Tensor of bboxes with shape (num_bboxes, 4)
            in [xmin, ymin, xmax, ymax] format.
        """
        whs = torch.from_numpy(self.bbox_whs).to(
            self.device, dtype=torch.float32)
        bboxes = bbox_cxcywh_to_xyxy(
            torch.cat([torch.zeros_like(whs), whs], dim=1))
        return bboxes

    def optimize(self):
        raise NotImplementedError

    def save_result(self, anchors, path=None):

        anchor_results = []
        start = 0
        for num in self.num_anchor_per_level:
            end = num + start
            anchor_results.append([(round(w), round(h))
                                   for w, h in anchors[start:end]])
            start = end

        self.logger.info(f'Anchor optimize result:{anchor_results}')
        if path:
            json_path = osp.join(path, 'anchor_optimize_result.json')
            dump(anchor_results, json_path)
            self.logger.info(f'Result saved in {json_path}')


class YOLOKMeansAnchorOptimizer(BaseAnchorOptimizer):
    r"""YOLO anchor optimizer using k-means. Code refer to `AlexeyAB/darknet.
    <https://github.com/AlexeyAB/darknet/blob/master/src/detector.c>`_.

    Args:
        iters (int): Maximum iterations for k-means.
    """

    def __init__(self, iters, **kwargs):

        super().__init__(**kwargs)
        self.iters = iters

    def optimize(self):
        anchors = self.kmeans_anchors()
        self.save_result(anchors, self.out_dir)

    def kmeans_anchors(self):
        self.logger.info(
            f'Start cluster {self.num_anchors} YOLO anchors with K-means...')
        bboxes = self.get_zero_center_bbox_tensor()
        cluster_center_idx = torch.randint(
            0, bboxes.shape[0], (self.num_anchors, )).to(self.device)

        assignments = torch.zeros((bboxes.shape[0], )).to(self.device)
        cluster_centers = bboxes[cluster_center_idx]
        if self.num_anchors == 1:
            cluster_centers = self.kmeans_maximization(bboxes, assignments,
                                                       cluster_centers)
            anchors = bbox_xyxy_to_cxcywh(cluster_centers)[:, 2:].cpu().numpy()
            anchors = sorted(anchors, key=lambda x: x[0] * x[1])
            return anchors

        prog_bar = ProgressBar(self.iters)
        for i in range(self.iters):
            converged, assignments = self.kmeans_expectation(
                bboxes, assignments, cluster_centers)
            if converged:
                self.logger.info(f'K-means process has converged at iter {i}.')
                break
            cluster_centers = self.kmeans_maximization(bboxes, assignments,
                                                       cluster_centers)
            prog_bar.update()
        print('\n')
        avg_iou = bbox_overlaps(bboxes,
                                cluster_centers).max(1)[0].mean().item()

        anchors = bbox_xyxy_to_cxcywh(cluster_centers)[:, 2:].cpu().numpy()
        anchors = sorted(anchors, key=lambda x: x[0] * x[1])
        self.logger.info(f'Anchor cluster finish. Average IOU: {avg_iou}')

        return anchors

    def kmeans_maximization(self, bboxes, assignments, centers):
        """Maximization part of EM algorithm(Expectation-Maximization)"""
        new_centers = torch.zeros_like(centers)
        for i in range(centers.shape[0]):
            mask = (assignments == i)
            if mask.sum():
                new_centers[i, :] = bboxes[mask].mean(0)
        return new_centers

    def kmeans_expectation(self, bboxes, assignments, centers):
        """Expectation part of EM algorithm(Expectation-Maximization)"""
        ious = bbox_overlaps(bboxes, centers)
        closest = ious.argmax(1)
        converged = (closest == assignments).all()
        return converged, closest


class YOLOV5KMeansAnchorOptimizer(BaseAnchorOptimizer):
    r"""YOLOv5 anchor optimizer using shape k-means.
    Code refer to `ultralytics/yolov5.
    <https://github.com/ultralytics/yolov5/blob/master/utils/autoanchor.py>`_.

    Args:
        iters (int): Maximum iterations for k-means.
        prior_match_thr (float): anchor-label width height
            ratio threshold hyperparameter.
    """

    def __init__(self,
                 iters,
                 prior_match_thr=4.0,
                 mutation_args=[0.9, 0.1],
                 augment_args=[0.9, 1.1],
                 **kwargs):

        super().__init__(**kwargs)
        self.iters = iters
        self.prior_match_thr = prior_match_thr
        [self.mutation_prob, self.mutation_sigma] = mutation_args
        [self.augment_min, self.augment_max] = augment_args

    def optimize(self):
        self.logger.info(
            f'Start cluster {self.num_anchors} YOLOv5 anchors with K-means...')

        bbox_whs = torch.from_numpy(self.bbox_whs).to(
            self.device, dtype=torch.float32)
        anchors = self.anchor_generate(
            bbox_whs,
            num=self.num_anchors,
            img_size=self.input_shape[0],
            prior_match_thr=self.prior_match_thr,
            iters=self.iters)
        best_ratio, mean_matched = self.anchor_metric(bbox_whs, anchors)
        self.logger.info(f'{mean_matched:.2f} anchors/target {best_ratio:.3f} '
                         'Best Possible Recall (BPR). ')
        self.save_result(anchors.tolist(), self.out_dir)

    def anchor_generate(self,
                        box_size: Tensor,
                        num: int = 9,
                        img_size: int = 640,
                        prior_match_thr: float = 4.0,
                        iters: int = 1000) -> Tensor:
        """cluster boxes metric with anchors.

        Args:
            box_size (Tensor): The size of the bxes, which shape is
                (box_num, 2),the number 2 means width and height.
            num (int): number of anchors.
            img_size (int): image size used for training
            prior_match_thr (float): width/height ratio threshold
                 used for training
            iters (int): iterations to evolve anchors using genetic algorithm

        Returns:
            anchors (Tensor): kmeans evolved anchors
        """

        thr = 1 / prior_match_thr

        # step1: filter small bbox
        box_size = self._filter_box(box_size)
        assert num <= len(box_size)

        # step2: init anchors
        if kmeans:
            try:
                self.logger.info(
                    'beginning init anchors with scipy kmeans method')
                # sigmas for whitening
                sigmas = box_size.std(0).cpu().numpy()
                anchors = kmeans(
                    box_size.cpu().numpy() / sigmas, num, iter=30)[0] * sigmas
                # kmeans may return fewer points than requested
                # if width/height is insufficient or too similar
                assert num == len(anchors)
            except Exception:
                self.logger.warning(
                    'scipy kmeans method cannot get enough points '
                    'because of width/height is insufficient or too similar, '
                    'now switching strategies from kmeans to random init.')
                anchors = np.sort(np.random.rand(num * 2)).reshape(
                    num, 2) * img_size
        else:
            self.logger.info(
                'cannot found scipy package, switching strategies from kmeans '
                'to random init, you can install scipy package to '
                'get better anchor init')
            anchors = np.sort(np.random.rand(num * 2)).reshape(num,
                                                               2) * img_size

        self.logger.info('init done, beginning evolve anchors...')
        # sort small to large
        anchors = torch.tensor(anchors[np.argsort(anchors.prod(1))]).to(
            box_size.device, dtype=torch.float32)

        # step3: evolve anchors use Genetic Algorithm
        prog_bar = ProgressBar(iters)
        fitness = self._anchor_fitness(box_size, anchors, thr)
        cluster_shape = anchors.shape

        for _ in range(iters):
            mutate_result = np.ones(cluster_shape)
            # mutate until a change occurs (prevent duplicates)
            while (mutate_result == 1).all():
                # mutate_result is scale factor of anchors, between 0.3 and 3
                mutate_result = (
                    (np.random.random(cluster_shape) < self.mutation_prob) *
                    random.random() * np.random.randn(*cluster_shape) *
                    self.mutation_sigma + 1).clip(0.3, 3.0)
            mutate_result = torch.from_numpy(mutate_result).to(box_size.device)
            new_anchors = (anchors.clone() * mutate_result).clip(min=2.0)
            new_fitness = self._anchor_fitness(box_size, new_anchors, thr)
            if new_fitness > fitness:
                fitness = new_fitness
                anchors = new_anchors.clone()

            prog_bar.update()
        print('\n')
        # sort small to large
        anchors = anchors[torch.argsort(anchors.prod(1))]
        self.logger.info(f'Anchor cluster finish. fitness = {fitness:.4f}')

        return anchors

    def anchor_metric(self,
                      box_size: Tensor,
                      anchors: Tensor,
                      threshold: float = 4.0) -> Tuple:
        """compute boxes metric with anchors.

        Args:
            box_size (Tensor): The size of the bxes, which shape
                is (box_num, 2), the number 2 means width and height.
            anchors (Tensor): The size of the bxes, which shape
                is (anchor_num, 2), the number 2 means width and height.
            threshold (float): the compare threshold of ratio

        Returns:
            Tuple: a tuple of metric result, best_ratio_mean and mean_matched
        """
        # step1: augment scale
        # According to the uniform distribution,the scaling scale between
        # augment_min and augment_max is randomly generated
        scale = np.random.uniform(
            self.augment_min, self.augment_max, size=(box_size.shape[0], 1))
        box_size = torch.tensor(
            np.array(
                [l[:, ] * s for s, l in zip(scale,
                                            box_size.cpu().numpy())])).to(
                                                box_size.device,
                                                dtype=torch.float32)
        # step2: calculate ratio
        min_ratio, best_ratio = self._metric(box_size, anchors)
        mean_matched = (min_ratio > 1 / threshold).float().sum(1).mean()
        best_ratio_mean = (best_ratio > 1 / threshold).float().mean()
        return best_ratio_mean, mean_matched

    def _filter_box(self, box_size: Tensor) -> Tensor:
        small_cnt = (box_size < 3.0).any(1).sum()
        if small_cnt:
            self.logger.warning(
                f'Extremely small objects found: {small_cnt} '
                f'of {len(box_size)} labels are <3 pixels in size')
        # filter > 2 pixels
        filter_sizes = box_size[(box_size >= 2.0).any(1)]
        return filter_sizes

    def _anchor_fitness(self, box_size: Tensor, anchors: Tensor, thr: float):
        """mutation fitness."""
        _, best = self._metric(box_size, anchors)
        return (best * (best > thr).float()).mean()

    def _metric(self, box_size: Tensor, anchors: Tensor) -> Tuple:
        """compute boxes metric with anchors.

        Args:
            box_size (Tensor): The size of the bxes, which shape is
                (box_num, 2), the number 2 means width and height.
            anchors (Tensor): The size of the bxes, which shape is
                (anchor_num, 2), the number 2 means width and height.

        Returns:
            Tuple: a tuple of metric result, min_ratio and best_ratio
        """

        # ratio means the (width_1/width_2 and height_1/height_2) ratio of each
        # box and anchor, the ratio shape is torch.Size([box_num,anchor_num,2])
        ratio = box_size[:, None] / anchors[None]

        # min_ratio records the min ratio of each box with all anchor,
        # min_ratio.shape is torch.Size([box_num,anchor_num])
        # notice:
        # smaller ratio means worse shape-match between boxes and anchors
        min_ratio = torch.min(ratio, 1 / ratio).min(2)[0]

        # find the best shape-match ratio for each box
        # box_best_ratio.shape is torch.Size([box_num])
        best_ratio = min_ratio.max(1)[0]

        return min_ratio, best_ratio


class YOLODEAnchorOptimizer(BaseAnchorOptimizer):
    """YOLO anchor optimizer using differential evolution algorithm.

    Args:
        iters (int): Maximum iterations for k-means.
        strategy (str): The differential evolution strategy to use.
            Should be one of:

                - 'best1bin'
                - 'best1exp'
                - 'rand1exp'
                - 'randtobest1exp'
                - 'currenttobest1exp'
                - 'best2exp'
                - 'rand2exp'
                - 'randtobest1bin'
                - 'currenttobest1bin'
                - 'best2bin'
                - 'rand2bin'
                - 'rand1bin'

            Default: 'best1bin'.
        population_size (int): Total population size of evolution algorithm.
            Default: 15.
        convergence_thr (float): Tolerance for convergence, the
            optimizing stops when ``np.std(pop) <= abs(convergence_thr)
            + convergence_thr * np.abs(np.mean(population_energies))``,
            respectively. Default: 0.0001.
        mutation (tuple[float]): Range of dithering randomly changes the
            mutation constant. Default: (0.5, 1).
        recombination (float): Recombination constant of crossover probability.
            Default: 0.7.
    """

    def __init__(self,
                 iters,
                 strategy='best1bin',
                 population_size=15,
                 convergence_thr=0.0001,
                 mutation=(0.5, 1),
                 recombination=0.7,
                 **kwargs):

        super().__init__(**kwargs)

        self.iters = iters
        self.strategy = strategy
        self.population_size = population_size
        self.convergence_thr = convergence_thr
        self.mutation = mutation
        self.recombination = recombination

    def optimize(self):
        anchors = self.differential_evolution()
        self.save_result(anchors, self.out_dir)

    def differential_evolution(self):
        bboxes = self.get_zero_center_bbox_tensor()

        bounds = []
        for i in range(self.num_anchors):
            bounds.extend([(0, self.input_shape[0]), (0, self.input_shape[1])])

        result = differential_evolution(
            func=self.avg_iou_cost,
            bounds=bounds,
            args=(bboxes, ),
            strategy=self.strategy,
            maxiter=self.iters,
            popsize=self.population_size,
            tol=self.convergence_thr,
            mutation=self.mutation,
            recombination=self.recombination,
            updating='immediate',
            disp=True)
        self.logger.info(
            f'Anchor evolution finish. Average IOU: {1 - result.fun}')
        anchors = [(w, h) for w, h in zip(result.x[::2], result.x[1::2])]
        anchors = sorted(anchors, key=lambda x: x[0] * x[1])
        return anchors

    @staticmethod
    def avg_iou_cost(anchor_params, bboxes):
        assert len(anchor_params) % 2 == 0
        anchor_whs = torch.tensor(
            [[w, h]
             for w, h in zip(anchor_params[::2], anchor_params[1::2])]).to(
                 bboxes.device, dtype=bboxes.dtype)
        anchor_boxes = bbox_cxcywh_to_xyxy(
            torch.cat([torch.zeros_like(anchor_whs), anchor_whs], dim=1))
        ious = bbox_overlaps(bboxes, anchor_boxes)
        max_ious, _ = ious.max(1)
        cost = 1 - max_ious.mean().item()
        return cost


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
