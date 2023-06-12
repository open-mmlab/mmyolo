_base_ = './yolov6_v3_m_syncbn_fast_8xb32-300e_widerface.py'

# ======================= Possible modified parameters =======================
# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 1
# The scaling factor that controls the width of the network structure
widen_factor = 1

# ============================== Unmodified in most cases ===================
model = dict(
    backbone=dict(
        use_cspsppf=False,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        block_cfg=dict(
            type='ConvWrapper',
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001)),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        block_cfg=dict(
            type='ConvWrapper',
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001)),
        block_act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(head_module=dict(reg_max=16, widen_factor=widen_factor)))
