_base_ = [
    '../_base_/base_instance-seg_static.py',
    '../../_base_/backends/tensorrt-fp16.py'
]

onnx_config = dict(input_shape=(1344, 800))
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 800, 1344],
                    opt_shape=[1, 3, 800, 1344],
                    max_shape=[1, 3, 800, 1344])))
    ])
