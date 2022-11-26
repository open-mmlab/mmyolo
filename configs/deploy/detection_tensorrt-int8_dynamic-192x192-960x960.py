_base_ = ['./base_dynamic.py']
backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=True, max_workspace_size=1 << 30, int8_mode=True),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 192, 192],
                    opt_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 960, 960])))
    ])
calib_config = dict(create_calib=True, calib_file='calib_data.h5')
use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # noqa E501
