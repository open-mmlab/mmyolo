_base_ = './yolov5_s-v61_fast_1xb96-100e_ionogram.py'

# ======================= Modified parameters =====================
# -----data related-----
train_batch_size_per_gpu = 32

# -----train val related-----
base_lr = _base_.base_lr * train_batch_size_per_gpu \
    / _base_.train_batch_size_per_gpu / 2
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]

# ===================== Unmodified in most cases ==================
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(dataset=dict(pipeline=train_pipeline)))

val_dataloader = dict(batch_size=train_batch_size_per_gpu)

test_dataloader = dict(batch_size=train_batch_size_per_gpu)

optim_wrapper = dict(optimizer=dict(lr=base_lr))
