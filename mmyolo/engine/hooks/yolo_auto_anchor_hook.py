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

    priority = 48

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def before_run(self, runner) -> None:
        """Create an ema copy of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(model.parameters()).device
        anchors = torch.tensor(
            runner.cfg.model.bbox_head.prior_generator.base_sizes,
            device=device)
        model.register_buffer('anchors', anchors)

    def before_train(self, runner: Runner) -> None:

        if runner.iter > 0:
            return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        print('begin reloading optimized anchors')

        rank, _ = get_dist_info()

        weights = model.state_dict()
        anchors_tensor = weights['anchors']
        if rank == 0 and not runner._has_loaded:
            runner_dataset = runner.train_dataloader.dataset
            self.optimizer.update(
                dataset=runner_dataset,
                device=runner_dataset[0]['inputs'].device,
                input_shape=runner.cfg['img_scale'],
                logger=MMLogger.get_current_instance())

            optimizer = TASK_UTILS.build(self.optimizer)
            anchors = optimizer.optimize()
            device = next(model.parameters()).device
            anchors_tensor = torch.tensor(anchors, device=device)

        broadcast(anchors_tensor)
        weights['anchors'] = anchors_tensor
        model.load_state_dict(weights)

        self.reinitialize(runner, model)

    def before_val(self, runner: Runner) -> None:

        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        print('begin reloading optimized anchors')
        self.reinitialize(runner, model)

    def before_test(self, runner: Runner) -> None:

        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        print('begin reloading optimized anchors')
        self.reinitialize(runner, model)

    def reinitialize(self, runner: Runner, model) -> None:
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
