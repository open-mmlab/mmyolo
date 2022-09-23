_base_ = ['../../mmdet/detection/detection_onnxruntime_static.py']
codebase_config = dict(
    type='mmyolo',
    task='ObjectDetection',
    module=['mmyolo.deploy.codebase.mmyolo'])
