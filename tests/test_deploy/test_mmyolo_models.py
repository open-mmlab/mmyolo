# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import numpy as np
import pytest
import torch
from mmengine import Config

try:
    import importlib
    importlib.import_module('mmdeploy')
except ImportError:
    pytest.skip('mmdeploy is not installed.', allow_module_level=True)

from mmdeploy.codebase import import_codebase
from mmdeploy.utils import Backend
from mmdeploy.utils.config_utils import register_codebase
from mmdeploy.utils.test import (WrapModel, check_backend, get_model_outputs,
                                 get_rewrite_outputs)

try:
    codebase = register_codebase('mmyolo')
    import_codebase(codebase, ['mmyolo.deploy'])
except ImportError:
    pytest.skip('mmyolo is not installed.', allow_module_level=True)


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False


def get_yolov5_head_model():
    """YOLOv5 Head Config."""
    test_cfg = Config(
        dict(
            multi_label=True,
            nms_pre=30000,
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.65),
            max_per_img=300))

    from mmyolo.models.dense_heads import YOLOv5Head
    head_module = dict(
        type='YOLOv5HeadModule',
        num_classes=4,
        in_channels=[2, 4, 8],
        featmap_strides=[8, 16, 32],
        num_base_priors=1)

    model = YOLOv5Head(head_module, test_cfg=test_cfg)

    model.requires_grad_(False)
    return model


@pytest.mark.parametrize('backend_type', [Backend.ONNXRUNTIME])
def test_yolov5_head_predict_by_feat(backend_type: Backend):
    """Test predict_by_feat rewrite of YOLOXHead."""
    check_backend(backend_type)
    yolov5_head = get_yolov5_head_model()
    yolov5_head.cpu().eval()
    s = 256
    batch_img_metas = [{
        'scale_factor': (1.0, 1.0),
        'pad_shape': (s, s, 3),
        'img_shape': (s, s, 3),
        'ori_shape': (s, s, 3)
    }]
    output_names = ['dets', 'labels']
    deploy_cfg = Config(
        dict(
            backend_config=dict(type=backend_type.value),
            onnx_config=dict(output_names=output_names, input_shape=None),
            codebase_config=dict(
                type='mmyolo',
                task='ObjectDetection',
                post_processing=dict(
                    score_threshold=0.05,
                    iou_threshold=0.5,
                    max_output_boxes_per_class=20,
                    pre_top_k=-1,
                    keep_top_k=10,
                    background_label_id=-1,
                ),
                module=['mmyolo.deploy'])))
    seed_everything(1234)
    cls_scores = [
        torch.rand(1, yolov5_head.num_classes * yolov5_head.num_base_priors,
                   4 * pow(2, i), 4 * pow(2, i)) for i in range(3, 0, -1)
    ]
    seed_everything(5678)
    bbox_preds = [
        torch.rand(1, 4 * yolov5_head.num_base_priors, 4 * pow(2, i),
                   4 * pow(2, i)) for i in range(3, 0, -1)
    ]
    seed_everything(9101)
    objectnesses = [
        torch.rand(1, 1 * yolov5_head.num_base_priors, 4 * pow(2, i),
                   4 * pow(2, i)) for i in range(3, 0, -1)
    ]

    # to get outputs of pytorch model
    model_inputs = {
        'cls_scores': cls_scores,
        'bbox_preds': bbox_preds,
        'objectnesses': objectnesses,
        'batch_img_metas': batch_img_metas,
        'with_nms': True
    }
    model_outputs = get_model_outputs(yolov5_head, 'predict_by_feat',
                                      model_inputs)

    # to get outputs of onnx model after rewrite
    wrapped_model = WrapModel(
        yolov5_head,
        'predict_by_feat',
        batch_img_metas=batch_img_metas,
        with_nms=True)
    rewrite_inputs = {
        'cls_scores': cls_scores,
        'bbox_preds': bbox_preds,
        'objectnesses': objectnesses,
    }
    rewrite_outputs, is_backend_output = get_rewrite_outputs(
        wrapped_model=wrapped_model,
        model_inputs=rewrite_inputs,
        deploy_cfg=deploy_cfg)

    if is_backend_output:
        # hard code to make two tensors with the same shape
        # rewrite and original codes applied different nms strategy
        min_shape = min(model_outputs[0].bboxes.shape[0],
                        rewrite_outputs[0].shape[1], 5)
        for i in range(len(model_outputs)):
            rewrite_outputs[0][i, :min_shape, 0::2] = \
                rewrite_outputs[0][i, :min_shape, 0::2].clamp_(0, s)
            rewrite_outputs[0][i, :min_shape, 1::2] = \
                rewrite_outputs[0][i, :min_shape, 1::2].clamp_(0, s)
            assert np.allclose(
                model_outputs[i].bboxes[:min_shape],
                rewrite_outputs[0][i, :min_shape, :4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].scores[:min_shape],
                rewrite_outputs[0][i, :min_shape, 4],
                rtol=1e-03,
                atol=1e-05)
            assert np.allclose(
                model_outputs[i].labels[:min_shape],
                rewrite_outputs[1][i, :min_shape],
                rtol=1e-03,
                atol=1e-05)
    else:
        assert rewrite_outputs is not None
