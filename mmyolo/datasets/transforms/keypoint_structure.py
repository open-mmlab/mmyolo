# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from torch import Tensor

DeviceType = Union[str, torch.device]
T = TypeVar('T')
IndexType = Union[slice, int, list, torch.LongTensor, torch.cuda.LongTensor,
                  torch.BoolTensor, torch.cuda.BoolTensor, np.ndarray]


class Keypoints(metaclass=ABCMeta):
    """The Keypoints class is for keypoints representation.

    Args:
        keypoints (Tensor or np.ndarray): The keypoint data with shape of
            (N, K, 2).
        keypoints_visible (Tensor or np.ndarray): The visibility of keypoints
            with shape of (N, K).
        device (str or torch.device, Optional): device of keypoints.
            Default to None.
        clone (bool): Whether clone ``keypoints`` or not. Defaults to True.
        flip_indices (list, Optional): The indices of keypoints when the
            images is flipped. Defaults to None.

    Notes:
        N: the number of instances.
        K: the number of keypoints.
    """

    def __init__(self,
                 keypoints: Union[Tensor, np.ndarray],
                 keypoints_visible: Union[Tensor, np.ndarray],
                 device: Optional[DeviceType] = None,
                 clone: bool = True,
                 flip_indices: Optional[List] = None) -> None:

        assert len(keypoints_visible) == len(keypoints)
        assert keypoints.ndim == 3
        assert keypoints_visible.ndim == 2

        keypoints = torch.as_tensor(keypoints)
        keypoints_visible = torch.as_tensor(keypoints_visible)

        if device is not None:
            keypoints = keypoints.to(device=device)
            keypoints_visible = keypoints_visible.to(device=device)

        if clone:
            keypoints = keypoints.clone()
            keypoints_visible = keypoints_visible.clone()

        self.keypoints = keypoints
        self.keypoints_visible = keypoints_visible
        self.flip_indices = flip_indices

    def flip_(self,
              img_shape: Tuple[int, int],
              direction: str = 'horizontal') -> None:
        """Flip boxes & kpts horizontally in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        """
        assert direction == 'horizontal'
        self.keypoints[..., 0] = img_shape[1] - self.keypoints[..., 0]
        self.keypoints = self.keypoints[:, self.flip_indices]
        self.keypoints_visible = self.keypoints_visible[:, self.flip_indices]

    def translate_(self, distances: Tuple[float, float]) -> None:
        """Translate boxes and keypoints in-place.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        assert len(distances) == 2
        distances = self.keypoints.new_tensor(distances).reshape(1, 1, 2)
        self.keypoints = self.keypoints + distances

    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        """Rescale boxes & keypoints w.r.t. rescale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
        """
        assert len(scale_factor) == 2

        scale_factor = self.keypoints.new_tensor(scale_factor).reshape(1, 1, 2)
        self.keypoints = self.keypoints * scale_factor

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Clip bounding boxes and set invisible keypoints outside the image
        boundary in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
        """

        kpt_outside = torch.logical_or(
            torch.logical_or(self.keypoints[..., 0] < 0,
                             self.keypoints[..., 1] < 0),
            torch.logical_or(self.keypoints[..., 0] > img_shape[1],
                             self.keypoints[..., 1] > img_shape[0]))
        self.keypoints_visible[kpt_outside] *= 0

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Geometrically transform bounding boxes and keypoints in-place using
        a homography matrix.

        Args:
            homography_matrix (Tensor or np.ndarray): A 3x3 tensor or ndarray
                representing the homography matrix for the transformation.
        """
        keypoints = self.keypoints
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = keypoints.new_tensor(homography_matrix)

        # Convert keypoints to homogeneous coordinates
        keypoints = torch.cat([
            self.keypoints,
            self.keypoints.new_ones(*self.keypoints.shape[:-1], 1)
        ],
                              dim=-1)

        # Transpose keypoints for matrix multiplication
        keypoints_T = torch.transpose(keypoints, -1, 0).contiguous().flatten(1)

        # Apply homography matrix to corners and keypoints
        keypoints_T = torch.matmul(homography_matrix, keypoints_T)

        # Transpose back to original shape
        keypoints_T = keypoints_T.reshape(3, self.keypoints.shape[1], -1)
        keypoints = torch.transpose(keypoints_T, -1, 0).contiguous()

        # Convert corners and keypoints back to non-homogeneous coordinates
        keypoints = keypoints[..., :2] / keypoints[..., 2:3]

        # Convert corners back to bounding boxes and update object attributes
        self.keypoints = keypoints

    @classmethod
    def cat(cls: Type[T], kps_list: Sequence[T], dim: int = 0) -> T:
        """Cancatenates an instance list into one single instance. Similar to
        ``torch.cat``.

        Args:
            box_list (Sequence[T]): A sequence of instances.
            dim (int): The dimension over which the box and keypoint are
                concatenated. Defaults to 0.

        Returns:
            T: Concatenated instance.
        """
        assert isinstance(kps_list, Sequence)
        if len(kps_list) == 0:
            raise ValueError('kps_list should not be a empty list.')

        assert dim == 0
        assert all(isinstance(keypoints, cls) for keypoints in kps_list)

        th_kpt_list = torch.cat(
            [keypoints.keypoints for keypoints in kps_list], dim=dim)
        th_kpt_vis_list = torch.cat(
            [keypoints.keypoints_visible for keypoints in kps_list], dim=dim)
        flip_indices = kps_list[0].flip_indices
        return cls(
            th_kpt_list,
            th_kpt_vis_list,
            clone=False,
            flip_indices=flip_indices)

    def __getitem__(self: T, index: IndexType) -> T:
        """Rewrite getitem to protect the last dimension shape."""
        if isinstance(index, np.ndarray):
            index = torch.as_tensor(index, device=self.device)
        if isinstance(index, Tensor) and index.dtype == torch.bool:
            assert index.dim() < self.keypoints.dim() - 1
        elif isinstance(index, tuple):
            assert len(index) < self.keypoints.dim() - 1
            # `Ellipsis`(...) is commonly used in index like [None, ...].
            # When `Ellipsis` is in index, it must be the last item.
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        keypoints = self.keypoints[index]
        keypoints_visible = self.keypoints_visible[index]
        if self.keypoints.dim() == 2:
            keypoints = keypoints.reshape(1, -1, 2)
            keypoints_visible = keypoints_visible.reshape(1, -1)
        return type(self)(
            keypoints,
            keypoints_visible,
            flip_indices=self.flip_indices,
            clone=False)

    def __repr__(self) -> str:
        """Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n' + str(self.keypoints) + ')'

    @property
    def num_keypoints(self) -> Tensor:
        """Compute the number of visible keypoints for each object."""
        return self.keypoints_visible.sum(dim=1).int()

    def __deepcopy__(self, memo):
        """Only clone the tensors when applying deepcopy."""
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other
        other.keypoints = self.keypoints.clone()
        other.keypoints_visible = self.keypoints_visible.clone()
        other.flip_indices = deepcopy(self.flip_indices)
        return other

    def clone(self: T) -> T:
        """Reload ``clone`` for tensors."""
        return type(self)(
            self.keypoints,
            self.keypoints_visible,
            flip_indices=self.flip_indices,
            clone=True)

    def to(self: T, *args, **kwargs) -> T:
        """Reload ``to`` for tensors."""
        return type(self)(
            self.keypoints.to(*args, **kwargs),
            self.keypoints_visible.to(*args, **kwargs),
            flip_indices=self.flip_indices,
            clone=False)

    @property
    def device(self) -> torch.device:
        """Reload ``device`` from self.tensor."""
        return self.keypoints.device
