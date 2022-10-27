_base_ = './ppyoloe_plus_s_fast_8xb8-80e_coco.py'

# TODO: training on ppyoloe need to be implemented.
train_batch_size_per_gpu = 32
max_epochs = 300

model = dict(
    data_preprocessor=dict(
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255., 0.224 * 255., 0.225 * 255.]),
    backbone=dict(block_cfg=dict(use_alpha=False)))
