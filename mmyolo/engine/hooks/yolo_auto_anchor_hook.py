# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dist import broadcast, get_dist_info
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.model import is_model_wrapper
from mmengine.registry import MODELS
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
        print('begin reloading optimized anchors')

        rank, _ = get_dist_info()
        device_w = next(model.parameters()).device
        anchors = torch.tensor(
            runner.cfg.model.bbox_head.prior_generator.base_sizes,
            device=device_w)
        model.register_buffer('anchors', anchors)

        weights = model.state_dict()
        key = 'anchors'
        anchors_tensor = weights[key]
        if rank == 0 and not runner._has_loaded:
            runner_dataset = runner.train_dataloader.dataset
            self.optimizer.update(
                dataset=runner_dataset,
                device=runner_dataset[0]['inputs'].device,
                input_shape=runner.cfg['img_scale'],
                logger=MMLogger.get_current_instance())

            optimizer = TASK_UTILS.build(self.optimizer)
            anchors = optimizer.optimize()
            anchors_tensor = torch.tensor(anchors, device=device_w)

        broadcast(anchors_tensor)
        weights[key] = anchors_tensor
        model.load_state_dict(weights)

        self.reinitialize_bbox_head(runner, model)
        runner.hooks[2].ema_model = MODELS.build(
            runner.hooks[2].ema_cfg, default_args=dict(model=model))

    def before_val(self, runner: Runner) -> None:

        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        print('begin reloading optimized anchors')
        self.reinitialize_bbox_head(runner, model)
        runner.hooks[2].ema_model = MODELS.build(
            runner.hooks[2].ema_cfg, default_args=dict(model=model))

    def before_test(self, runner: Runner) -> None:

        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        print('begin reloading optimized anchors')
        self.reinitialize_bbox_head(runner, model)
        runner.hooks[2].ema_model = MODELS.build(
            runner.hooks[2].ema_cfg, default_args=dict(model=model))

    def reinitialize_bbox_head(self, runner: Runner, model) -> None:
        anchors_tensor = model.state_dict()['anchors']
        base_sizes = anchors_tensor.tolist()
        device = anchors_tensor.device
        prior_generator = runner.cfg.model.bbox_head.prior_generator
        prior_generator.update(base_sizes=base_sizes)

        model.bbox_head.prior_generator = TASK_UTILS.build(prior_generator)

        priors_base_sizes = torch.tensor(
            base_sizes, dtype=torch.float, device=device)
        featmap_strides = torch.tensor(
            model.bbox_head.featmap_strides, dtype=torch.float,
            device=device)[:, None, None]
        model.bbox_head.priors_base_sizes = priors_base_sizes / featmap_strides
