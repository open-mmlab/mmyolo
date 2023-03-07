_base_ = 'yolov5_s-v61_fast_1xb12-40e_cat.py'

model = dict(
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='YOLOXBatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=1)
        ]))
