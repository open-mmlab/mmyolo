_base_ = './yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

# ========================modified parameters======================
data_root = 'data/balloon/'
# Path of train annotation file
train_ann_file = 'train.json'
train_data_prefix = 'train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'val.json'
val_data_prefix = 'val/'  # Prefix of val image path
metainfo = {
    'classes': ('balloon', ),
    'palette': [
        (220, 20, 60),
    ]
}
num_classes = 1

train_batch_size_per_gpu = 4
train_num_workers = 2
log_interval = 1

# =======================Unmodified in most cases==================
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=train_data_prefix),
        ann_file=train_ann_file))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + val_ann_file)
test_evaluator = val_evaluator
model = dict(bbox_head=dict(head_module=dict(num_classes=num_classes)))
default_hooks = dict(logger=dict(interval=log_interval))
