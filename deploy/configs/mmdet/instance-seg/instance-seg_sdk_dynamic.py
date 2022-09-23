_base_ = ['../_base_/base_dynamic.py', '../../_base_/backends/sdk.py']

codebase_config = dict(model_type='sdk', has_mask=True)

backend_config = dict(pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
])
