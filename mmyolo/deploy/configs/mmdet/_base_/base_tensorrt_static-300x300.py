_base_ = ['./base_static.py', '../../_base_/backends/tensorrt.py']

onnx_config = dict(input_shape=(300, 300))

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 300, 300],
                    opt_shape=[1, 3, 300, 300],
                    max_shape=[1, 3, 300, 300])))
    ])
