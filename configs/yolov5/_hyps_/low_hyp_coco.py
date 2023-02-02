# parameters that often need to be modified
data_hyp = dict(
    data_root='data/coco/',
    dataset_type='YOLOv5CocoDataset',
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/'),
    num_classes=80,
    img_scale=(640, 640),  # width, height
    train_batch_size_per_gpu=16,
    train_num_workers=8,
    val_batch_size_per_gpu=1,
    val_num_workers=2,
    # persistent_workers must be False if num_workers is 0.
    persistent_workers=True
)

model_hyp = dict(
    anchors=[
        [(10, 13), (16, 30), (33, 23)],  # P3/8
        [(30, 61), (62, 45), (59, 119)],  # P4/16
        [(116, 90), (156, 198), (373, 326)]  # P5/32
    ],
    deepen_factor=0.33,
    widen_factor=0.5
)

train_val_hyp = dict(
    # Base learning rate for optim_wrapper
    base_lr=0.01,
    max_epochs=300,
    save_epoch_intervals=10,
    # only on Val
    batch_shapes_cfg=dict(
        type='BatchShapePolicy',
        batch_size=data_hyp.val_batch_size_per_gpu,
        img_size=data_hyp.img_scale[0],
        size_divisor=32,
        extra_pad_ratio=0.5)
)

# single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)
