# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from mmdet.structures.bbox.transforms import get_box_tensor
from torch import Tensor


def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x


def gt_instances_preprocess(batch_gt_instances: Union[Tensor, Sequence],
                            batch_size: int) -> Tensor:
    """Split batch_gt_instances with batch size.

    From [all_gt_bboxes, box_dim+2] to [batch_size, number_gt, box_dim+1].
    For horizontal box, box_dim=4, for rotated box, box_dim=5

    If some shape of single batch smaller than
    gt bbox len, then using zeros to fill.

    Args:
        batch_gt_instances (Sequence[Tensor]): Ground truth
            instances for whole batch, shape [all_gt_bboxes, box_dim+2]
        batch_size (int): Batch size.

    Returns:
        Tensor: batch gt instances data, shape
                [batch_size, number_gt, box_dim+1]
    """
    if isinstance(batch_gt_instances, Sequence):
        max_gt_bbox_len = max(
            [len(gt_instances) for gt_instances in batch_gt_instances])
        # fill zeros with length box_dim+1 if some shape of
        # single batch not equal max_gt_bbox_len
        batch_instance_list = []
        for index, gt_instance in enumerate(batch_gt_instances):
            bboxes = gt_instance.bboxes
            labels = gt_instance.labels
            box_dim = get_box_tensor(bboxes).size(-1)
            batch_instance_list.append(
                torch.cat((labels[:, None], bboxes), dim=-1))

            if bboxes.shape[0] >= max_gt_bbox_len:
                continue

            fill_tensor = bboxes.new_full(
                [max_gt_bbox_len - bboxes.shape[0], box_dim + 1], 0)
            batch_instance_list[index] = torch.cat(
                (batch_instance_list[index], fill_tensor), dim=0)

        return torch.stack(batch_instance_list)
    else:
        # faster version
        # format of batch_gt_instances: [img_ind, cls_ind, (box)]
        # For example horizontal box should be:
        # [img_ind, cls_ind, x1, y1, x2, y2]
        # Rotated box should be
        # [img_ind, cls_ind, x, y, w, h, a]

        # sqlit batch gt instance [all_gt_bboxes, box_dim+2] ->
        # [batch_size, max_gt_bbox_len, box_dim+1]
        assert isinstance(batch_gt_instances, Tensor)
        box_dim = batch_gt_instances.size(-1) - 2
        if len(batch_gt_instances) > 0:
            gt_images_indexes = batch_gt_instances[:, 0]
            max_gt_bbox_len = gt_images_indexes.unique(
                return_counts=True)[1].max()
            # fill zeros with length box_dim+1 if some shape of
            # single batch not equal max_gt_bbox_len
            batch_instance = torch.zeros(
                (batch_size, max_gt_bbox_len, box_dim + 1),
                dtype=batch_gt_instances.dtype,
                device=batch_gt_instances.device)

            for i in range(batch_size):
                match_indexes = gt_images_indexes == i
                gt_num = match_indexes.sum()
                if gt_num:
                    batch_instance[i, :gt_num] = batch_gt_instances[
                        match_indexes, 1:]
        else:
            batch_instance = torch.zeros((batch_size, 0, box_dim + 1),
                                         dtype=batch_gt_instances.dtype,
                                         device=batch_gt_instances.device)

        return batch_instance


class OutputSaveObjectWrapper:
    """A wrapper class that saves the output of function calls on an object."""

    def __init__(self, obj: Any) -> None:
        self.obj = obj
        self.log = defaultdict(list)

    def __getattr__(self, attr: str) -> Any:
        """Overrides the default behavior when an attribute is accessed.

        - If the attribute is callable, hooks the attribute and saves the
        returned value of the function call to the log.
        - If the attribute is not callable, saves the attribute's value to the
        log and returns the value.
        """
        orig_attr = getattr(self.obj, attr)

        if not callable(orig_attr):
            self.log[attr].append(orig_attr)
            return orig_attr

        def hooked(*args: Tuple, **kwargs: Dict) -> Any:
            """The hooked function that logs the return value of the original
            function."""
            result = orig_attr(*args, **kwargs)
            self.log[attr].append(result)
            return result

        return hooked

    def clear(self):
        """Clears the log of function call outputs."""
        self.log.clear()

    def __deepcopy__(self, memo):
        """Only copy the object when applying deepcopy."""
        other = type(self)(deepcopy(self.obj))
        memo[id(self)] = other
        return other


class OutputSaveFunctionWrapper:
    """A class that wraps a function and saves its outputs.

    This class can be used to decorate a function to save its outputs. It wraps
    the function with a `__call__` method that calls the original function and
    saves the results in a log attribute.
    Args:
        func (Callable): A function to wrap.
        spec (Optional[Dict]): A dictionary of global variables to use as the
            namespace for the wrapper. If `None`, the global namespace of the
            original function is used.
    """

    def __init__(self, func: Callable, spec: Optional[Dict]) -> None:
        """Initializes the OutputSaveFunctionWrapper instance."""
        assert callable(func)
        self.log = []
        self.func = func
        self.func_name = func.__name__

        if isinstance(spec, dict):
            self.spec = spec
        elif hasattr(func, '__globals__'):
            self.spec = func.__globals__
        else:
            raise ValueError

    def __call__(self, *args, **kwargs) -> Any:
        """Calls the wrapped function with the given arguments and saves the
        results in the `log` attribute."""
        results = self.func(*args, **kwargs)
        self.log.append(results)
        return results

    def __enter__(self) -> None:
        """Enters the context and sets the wrapped function to be a global
        variable in the specified namespace."""
        self.spec[self.func_name] = self
        return self.log

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exits the context and resets the wrapped function to its original
        value in the specified namespace."""
        self.spec[self.func_name] = self.func
