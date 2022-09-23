_base_ = ['./base_dynamic.py', '../../_base_/backends/tensorrt.py']

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 300, 300],
                    opt_shape=[1, 3, 300, 300],
                    max_shape=[1, 3, 512, 512])))
    ])
