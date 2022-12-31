_base_ = './ppyoloe_plus_s_fast_8xb8-80e_coco.py'

train_batch_size_per_gpu = 32
max_epochs = 300

# Base learning rate for optim_wrapper
base_lr = 0.01

model = dict(
    data_preprocessor=dict(
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255., 0.224 * 255., 0.225 * 255.]),
    backbone=dict(block_cfg=dict(use_alpha=False)),
    train_cfg=dict(initial_epoch=100))

train_dataloader = dict(batch_size=train_batch_size_per_gpu)

optim_wrapper = dict(optimizer=dict(lr=base_lr))

default_hooks = dict(param_scheduler=dict(total_epochs=int(max_epochs * 1.2)))

train_cfg = dict(max_epochs=max_epochs)

# The pretrained model is converted from official PPYOLOE
load_from = 'https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_pretrain/cspresnet_s_imagenet1k_pretrained-2be81763.pth'  # noqa
