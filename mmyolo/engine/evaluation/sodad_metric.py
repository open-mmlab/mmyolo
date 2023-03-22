# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import itertools
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

from mmengine.logging import MMLogger
from mmengine.fileio import load
from terminaltables import AsciiTable
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.evaluation import CocoMetric
from mmyolo.registry import METRICS
from .sodad_eval import SODADeval


@METRICS.register_module()
class SodadMetric(CocoMetric):
    """SODAD evaluation metric.
    """
    default_prefix: Optional[str] = 'sodad'

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return eval_results

        for metric in self.metrics:
            logger.info(f'Evaluating {metric}...')

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    preds, self.proposal_nums, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error(
                    'The testing results of the whole dataset is empty.')
                break

            SODADEval = SODADeval(self._coco_api, coco_dt, iou_type)
            SODADEval.params.catIds = self.cat_ids
            SODADEval.params.imgIds = self.img_ids
            SODADEval.params.maxDets = list(self.proposal_nums)
            SODADEval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            tod_metric_names = {
                'AP': 0,
                'AP_50': 1,
                'AP_75': 2,
                'AP_eS': 3,
                'AP_rS': 4,
                'AP_gS': 5,
                'AP_Normal': 6,
                'AR@100': 7,
                'AR@300': 8,
                'AR@1000': 9,
                'AR_eS@1000': 10,
                'AR_rS@1000': 11,
                'AR_gS@1000': 12,
                'AR_Normal@1000': 13
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in tod_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                SODADEval.params.useCats = 0
                SODADEval.evaluate()
                SODADEval.accumulate()
                SODADEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AP', 'AP_50', 'AP_75', 'AP_eS', 'AP_rS', 'AP_gS',
                        'AP_Normal'
                    ]

                for item in metric_items:
                    val = float(
                        f'{SODADEval.stats[tod_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                SODADEval.evaluate()
                SODADEval.accumulate()
                SODADEval.summarize()
                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = SODADEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{round(ap, 3)}'))
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info('\n' + table.table)

                if metric_items is None:
                    metric_items = [
                        'AP', 'AP_50', 'AP_75', 'AP_eS', 'AP_rS', 'AP_gS',
                        'AP_Normal'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = SODADEval.stats[tod_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                ap = SODADEval.stats[:7]
                logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                            f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                            f'{ap[4]:.3f} {ap[5]:.3f} {ap[6]:.3f}')

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
