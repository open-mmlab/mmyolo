_base_ = './yolov6_s_fast_1xb32-100e_ionogram.py'

# ======================= Modified parameters =====================
base_lr = _base_.base_lr * 4
optim_wrapper = dict(optimizer=dict(lr=base_lr))
max_epochs = 200
load_from = None

# ==================== Unmodified in most cases ===================
train_cfg = dict(
    max_epochs=max_epochs,
    val_begin=20,
)

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=50))
