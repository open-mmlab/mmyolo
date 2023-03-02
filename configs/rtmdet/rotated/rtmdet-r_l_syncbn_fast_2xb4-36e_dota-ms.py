_base_ = './rtmdet-r_l_syncbn_fast_2xb4-36e_dota.py'

# ========================modified parameters======================
data_root = 'data/split_ms_dota/'
# Path of test images folder
test_data_prefix = 'test/images/'
# Submission dir for result submit
submission_dir = './work_dirs/{{fileBasenameNoExtension}}/submission'

# =======================Unmodified in most cases==================
train_dataloader = dict(dataset=dict(data_root=data_root))

val_dataloader = dict(dataset=dict(data_root=data_root))

# Inference on val dataset
test_dataloader = val_dataloader

# Inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
# test_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         ann_file='', # test set has no annotation
#         data_prefix=dict(img_path=test_data_prefix),
#         pipeline=_base_.test_pipeline))
# test_evaluator = dict(
#     type='mmrotate.DOTAMetric',
#     format_only=True,
#     merge_patches=True,
#     outfile_prefix=submission_dir)
