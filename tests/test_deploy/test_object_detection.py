# Copyright (c) OpenMMLab. All rights reserved.
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pytest
import torch
from mmengine import Config

try:
    import importlib
    importlib.import_module('mmdeploy')
except ImportError:
    pytest.skip('mmdeploy is not installed.', allow_module_level=True)

import mmdeploy.backend.onnxruntime as ort_apis
from mmdeploy.apis import build_task_processor
from mmdeploy.codebase import import_codebase
from mmdeploy.utils import load_config
from mmdeploy.utils.config_utils import register_codebase
from mmdeploy.utils.test import SwitchBackendWrapper

try:
    codebase = register_codebase('mmyolo')
    import_codebase(codebase, ['mmyolo.deploy'])
except ImportError:
    pytest.skip('mmyolo is not installed.', allow_module_level=True)

model_cfg_path = 'tests/test_deploy/data/model.py'
model_cfg = load_config(model_cfg_path)[0]
model_cfg.test_dataloader.dataset.data_root = \
    'tests/data'
model_cfg.test_dataloader.dataset.ann_file = 'coco_sample.json'
model_cfg.test_evaluator.ann_file = \
    'tests/coco_sample.json'
deploy_cfg = Config(
    dict(
        backend_config=dict(type='onnxruntime'),
        codebase_config=dict(
            type='mmyolo',
            task='ObjectDetection',
            post_processing=dict(
                score_threshold=0.05,
                confidence_threshold=0.005,  # for YOLOv3
                iou_threshold=0.5,
                max_output_boxes_per_class=200,
                pre_top_k=5000,
                keep_top_k=100,
                background_label_id=-1,
            ),
            module=['mmyolo.deploy']),
        onnx_config=dict(
            type='onnx',
            export_params=True,
            keep_initializers_as_inputs=False,
            opset_version=11,
            input_shape=None,
            input_names=['input'],
            output_names=['dets', 'labels'])))
onnx_file = NamedTemporaryFile(suffix='.onnx').name
task_processor = None
img_shape = (32, 32)
img = np.random.rand(*img_shape, 3)


@pytest.fixture(autouse=True)
def init_task_processor():
    global task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')


@pytest.fixture
def backend_model():
    from mmdeploy.backend.onnxruntime import ORTWrapper
    ort_apis.__dict__.update({'ORTWrapper': ORTWrapper})
    wrapper = SwitchBackendWrapper(ORTWrapper)
    wrapper.set(
        outputs={
            'dets': torch.rand(1, 10, 5).sort(2).values,
            'labels': torch.randint(0, 10, (1, 10))
        })

    yield task_processor.build_backend_model([''])

    wrapper.recover()


def test_visualize(backend_model):
    img_path = 'tests/data/color.jpg'
    input_dict, _ = task_processor.create_input(
        img_path, input_shape=img_shape)
    results = backend_model.test_step(input_dict)[0]
    with TemporaryDirectory() as dir:
        filename = dir + 'tmp.jpg'
        task_processor.visualize(img, results, filename, 'window')
        assert os.path.exists(filename)
