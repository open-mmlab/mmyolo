_base_ = './ppyoloe_s_fast_8xb32-300e_coco.py'

max_epochs = 400

model = dict(train_cfg=dict(initial_epoch=133))

default_hooks = dict(param_scheduler=dict(total_epochs=int(max_epochs * 1.2)))

train_cfg = dict(max_epochs=max_epochs)
