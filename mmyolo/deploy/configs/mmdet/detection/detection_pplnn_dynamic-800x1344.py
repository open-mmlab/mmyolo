_base_ = ['../_base_/base_dynamic.py', '../../_base_/backends/pplnn.py']

onnx_config = dict(input_shape=None)

backend_config = dict(model_inputs=dict(opt_shape=[1, 3, 800, 1344]))
