_base_ = ['./tensorrt.py']

backend_config = dict(common_config=dict(fp16_mode=True, int8_mode=True))

calib_config = dict(create_calib=True, calib_file='calib_data.h5')
