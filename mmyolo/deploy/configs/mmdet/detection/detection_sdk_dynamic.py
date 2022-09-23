_base_ = ['../_base_/base_dynamic.py', '../../_base_/backends/sdk.py']

codebase_config = dict(model_type='sdk')

backend_config = dict(pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
])
