_base_ = [
    '../../mmdet/detection/detection_tensorrt-int8_dynamic-320x320-1344x1344.py'  # noqa: E501
]
codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection',
    module=[
        'mmyolo.deploy.codebase.mmyolo', 'mmyolo.datasets', 'mmyolo.models',
        'mmyolo.engine'
    ],
    extra_dependent_library=['mmdet'])
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 640, 640])))
    ])
