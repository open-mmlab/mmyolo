_base_ = './yolov5_s-v61_fast_1xb96-100e_ionogram.py'

# ======================= Modified parameters =====================
# -----train val related-----
base_lr = _base_.base_lr * 4
max_epochs = 200
load_from = None
logger_interval = 50

train_cfg = dict(max_epochs=max_epochs, )

# ===================== Unmodified in most cases ==================
optim_wrapper = dict(optimizer=dict(lr=base_lr))

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=logger_interval))
