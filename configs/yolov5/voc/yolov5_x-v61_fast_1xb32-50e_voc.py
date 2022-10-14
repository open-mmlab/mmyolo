_base_ = './yolov5_s-v61_fast_1xb64-50e_voc.py'

deepen_factor = 1.33
widen_factor = 1.25
train_batch_size_per_gpu = 32
train_num_workers = 8

# TODO: need to add pretrained_model
load_from = None

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu, num_workers=train_num_workers)

optim_wrapper = dict(
    optimizer=dict(batch_size_per_gpu=train_batch_size_per_gpu))
