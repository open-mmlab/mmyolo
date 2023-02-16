import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import FileClient, dump, load
from mmengine.logging import MMLogger
from mmpose.evaluation.functional import oks_nms, soft_oks_nms
from mmpose.evaluation.metrics.coco_metric import CocoMetric as MMPosCocoMetric
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from terminaltables import AsciiTable
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval


from mmyolo.registry import METRICS


@METRICS.register_module()
class CocoMetric(MMPosCocoMetric):

    @property
    def dataset_meta(self) -> Optional[dict]:
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta) -> Optional[dict]:
        self._dataset_meta = parse_pose_metainfo(dict(from_file="../configs/coco.py"))

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model, each of which has the following keys:

                - 'id': The id of the sample
                - 'img_id': The image_id of the sample
                - 'pred_instances': The prediction results of instance(s)
        """
        for data_sample in data_samples:
            if 'pred_instances' not in data_sample:
                raise ValueError(
                    '`pred_instances` are required to process the '
                    f'predictions results in {self.__class__.__name__}. ')

            # keypoints.shape: [N, K, 2],
            # N: number of instances, K: number of keypoints
            # for topdown-style output, N is usually 1, while for
            # bottomup-style output, N is the number of instances in the image
            keypoints = data_sample['pred_instances']['keypoints']
            # [N, K], the scores for all keypoints of all instances
            keypoint_scores = data_sample['pred_instances']['keypoint_scores']
            scores = data_sample['pred_instances']['scores']
            assert keypoint_scores.shape == keypoints.shape[:2]

            result = dict()
            # result['id'] = data_sample['id']
            result['img_id'] = data_sample['img_id']
            result['keypoints'] = keypoints
            result['keypoint_scores'] = keypoint_scores
            result['bbox_scores'] = scores

            # get area information
            if 'bbox_scales' in data_sample['gt_instances']:
                result['areas'] = np.prod(
                    data_sample['gt_instances']['bbox_scales'], axis=1)
            # add converted result to the results list
            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        kpts = defaultdict(list)

        # group the results by img_id
        for result in results:
            img_id = result['img_id']
            for idx in range(len(result['keypoints'])):
                instance = {
                    # 'id': result['id'],
                    'img_id': result['img_id'],
                    'keypoints': result['keypoints'][idx],
                    'keypoint_scores': result['keypoint_scores'][idx],
                    'bbox_score': result['bbox_scores'][idx],
                }

                if 'areas' in result:
                    instance['area'] = result['areas'][idx]
                else:
                    # use keypoint to calculate bbox and get area
                    keypoints = result['keypoints'][idx].numpy()
                    area = (
                        np.max(keypoints[:, 0]) - np.min(keypoints[:, 0])) * (
                            np.max(keypoints[:, 1]) - np.min(keypoints[:, 1]))
                    instance['area'] = area

                kpts[img_id].append(instance)

        # sort keypoint results according to id and remove duplicate ones
        kpts = self._sort_and_unique_bboxes(kpts, key='img_id')

        # score the prediction results according to `score_mode`
        # and perform NMS according to `nms_mode`
        valid_kpts = defaultdict(list)
        num_keypoints = self.dataset_meta['num_keypoints']
        for img_id, instances in kpts.items():
            for instance in instances:
                # concatenate the keypoint coordinates and scores
                instance['keypoints'] = np.concatenate([
                    instance['keypoints'], instance['keypoint_scores'][:, None]
                ],
                                                       axis=-1)
                if self.score_mode == 'bbox':
                    instance['score'] = instance['bbox_score']
                else:
                    bbox_score = instance['bbox_score']
                    if self.score_mode == 'bbox_rle':
                        keypoint_scores = instance['keypoint_scores']
                        instance['score'] = float(bbox_score +
                                                  np.mean(keypoint_scores) +
                                                  np.max(keypoint_scores))

                    else:  # self.score_mode == 'bbox_keypoint':
                        mean_kpt_score = 0
                        valid_num = 0
                        for kpt_idx in range(num_keypoints):
                            kpt_score = instance['keypoint_scores'][kpt_idx]
                            if kpt_score > self.keypoint_score_thr:
                                mean_kpt_score += kpt_score
                                valid_num += 1
                        if valid_num != 0:
                            mean_kpt_score /= valid_num
                        instance['score'] = bbox_score * mean_kpt_score
            # perform nms
            if self.nms_mode == 'none':
                valid_kpts[img_id] = instances
            else:
                nms = oks_nms if self.nms_mode == 'oks_nms' else soft_oks_nms
                keep = nms(
                    instances,
                    self.nms_thr,
                    sigmas=self.dataset_meta['sigmas'])
                valid_kpts[img_id] = [instances[_keep] for _keep in keep]

        # convert results to coco style and dump into a json file
        self.results2json(valid_kpts, outfile_prefix=outfile_prefix)

        # only format the results without doing quantitative evaluation
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return {}

        # evaluation results
        eval_results = OrderedDict()
        logger.info(f'Evaluating {self.__class__.__name__}...')
        info_str = self._do_python_keypoint_eval(outfile_prefix)
        name_value = OrderedDict(info_str)
        eval_results.update(name_value)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results