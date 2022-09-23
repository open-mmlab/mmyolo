# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable

from mmengine.registry import Registry
from mmengine import Config

from mmdeploy.codebase.base import CODEBASE, MMCodebase
from mmdeploy.codebase.mmdet.deploy import ObjectDetection
from mmdeploy.utils import Codebase, Task

MMYOLO_TASK = Registry('mmyolo_tasks')


@CODEBASE.register_module(Codebase.MMYOLO.value)
class MMYolo(MMCodebase):
    """MMYolo codebase class."""

    task_registry = MMYOLO_TASK


def _get_dataset_metainfo(model_cfg: Config):
    """Get metainfo of dataset.

    Args:
        model_cfg Config: Input model Config object.

    Returns:
        list[str]: A list of string specifying names of different class.
    """
    from mmyolo import datasets  # noqa
    from mmyolo.registry import DATASETS

    module_dict = DATASETS.module_dict
    for dataloader_name in [
            'test_dataloader', 'val_dataloader', 'train_dataloader'
    ]:
        if dataloader_name not in model_cfg:
            continue
        dataloader_cfg = model_cfg[dataloader_name]
        dataset_cfg = dataloader_cfg.dataset
        dataset_cls = module_dict.get(dataset_cfg.type, None)
        if dataset_cls is None:
            continue
        if hasattr(dataset_cls, '_load_metainfo') and isinstance(
                dataset_cls._load_metainfo, Callable):
            meta = dataset_cls._load_metainfo(
                dataset_cfg.get('metainfo', None))
            if meta is not None:
                return meta
        if hasattr(dataset_cls, 'METAINFO'):
            return dataset_cls.METAINFO

    return None


@MMYOLO_TASK.register_module(Task.OBJECT_DETECTION.value)
class YoloObjectDetection(ObjectDetection):

    def get_visualizer(self, name: str, save_dir: str):
        from mmdet.visualization import DetLocalVisualizer  # noqa: F401,F403
        metainfo = _get_dataset_metainfo(self.model_cfg)
        visualizer = super().get_visualizer(name, save_dir)
        if metainfo is not None:
            visualizer.dataset_meta = metainfo
        return visualizer
