# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dist import broadcast, get_dist_info
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmyolo.registry import HOOKS, TASK_UTILS


@HOOKS.register_module()
class YOLOAutoAnchorHook(Hook):

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def before_train(self, runner: Runner) -> None:

        if runner.iter > 0:
            return

        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        rank, _ = get_dist_info()

        weights = model.state_dict()
        key = 'bbox_head.prior_generator.anchors'
        anchors_tensor = weights[key]
        device_m = weights[key].device
        if rank == 0 and not runner._has_loaded:
            runner_dataset = runner.train_dataloader.dataset
            self.optimizer.update(
                dataset=runner_dataset,
                device=runner_dataset[0]['inputs'].device,
                input_shape=runner.cfg['img_scale'],
                logger=MMLogger.get_current_instance())

            optimizer = TASK_UTILS.build(self.optimizer)
            anchors = optimizer.optimize()
            anchors_tensor = torch.tensor(anchors, device=device_m)

        broadcast(anchors_tensor)
        weights[key] = anchors_tensor
        model.load_state_dict(weights)
        self.reinitialize_bbox_head(runner, model, device_m)

    def before_val(self, runner: Runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        prior_generator = model.bbox_head.prior_generator
        device = prior_generator.anchors.device
        self.reinitialize_bbox_head(runner, model, device)

    def before_test(self, runner: Runner) -> None:

        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        prior_generator = model.bbox_head.prior_generator
        device = prior_generator.anchors.device
        self.reinitialize_bbox_head(runner, model, device)

    def reinitialize_bbox_head(self, runner: Runner, model, device) -> None:
        priors_base_sizes = torch.tensor(
            model.bbox_head.prior_generator.base_sizes,
            dtype=torch.float,
            device=device)
        featmap_strides = torch.tensor(
            model.bbox_head.featmap_strides, dtype=torch.float,
            device=device)[:, None, None]
        model.bbox_head.priors_base_sizes = priors_base_sizes / featmap_strides
