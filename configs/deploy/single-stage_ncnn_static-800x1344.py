_base_ = '../_base_/base_static.py'

codebase_config = dict(model_type='ncnn_end2end')
onnx_config = dict(output_names=['detection_output'], input_shape=[1344, 800])
