_base_ = '../_base_/base_static.py'
backend_config = dict(type='ncnn', precision='FP32', use_vulkan=False)
codebase_config = dict(model_type='ncnn_end2end')
onnx_config = dict(output_names=['detection_output'], input_shape=[300, 300])
