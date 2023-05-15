# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import to_tensor
from mmdet.datasets.transforms import PackDetInputs as MMDET_PackDetInputs
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData, PixelData

from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackDetInputs(MMDET_PackDetInputs):
    """Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    Compared to mmdet, we just add the `gt_panoptic_seg` field and logic.
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_keypoints': 'keypoints',
        'gt_keypoints_visible': 'keypoints_visible'
    }

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.
        Args:
            results (dict): Result dict from the data pipeline.
        Returns:
            dict:
            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = img

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]
        if 'gt_keypoints' in results:
            results['gt_keypoints_visible'] = results[
                'gt_keypoints'].keypoints_visible
            results['gt_keypoints'] = results['gt_keypoints'].keypoints

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        # In order to unify the support for the overlap mask annotations
        # i.e. mask overlap annotations in (h,w) format,
        # we use the gt_panoptic_seg field to unify the modeling
        if 'gt_panoptic_seg' in results:
            data_sample.gt_panoptic_seg = PixelData(
                pan_seg=results['gt_panoptic_seg'])

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                                   f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results
