_base_ = ['./base_static.py']
onnx_config = dict(
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'dets': {
            0: 'batch',
            1: 'num_dets'
        },
        'labels': {
            0: 'batch',
            1: 'num_dets'
        }
    })
