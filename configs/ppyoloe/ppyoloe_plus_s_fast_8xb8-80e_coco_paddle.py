_base_ = './ppyoloe_plus_s_fast_8xb8-80e_coco_paddle.py'

# dataset settings
dataset_type = 'PPYOLOECocoDataset'

model = dict(
    data_preprocessor=dict(
        batch_augments=[
            dict(
                type='PPYOLOEBatchSyncRandomResizeallopencv',
                random_size_range=(320, 800),
                interval=1,
                size_divisor=32,
                random_interp=True,
                keep_ratio=False)
        ],
        bgr_to_rgb=True))

train_dataloader = dict(dataset=dict(type=dataset_type))

val_dataloader = dict(dataset=dict(type=dataset_type))

test_dataloader = val_dataloader
