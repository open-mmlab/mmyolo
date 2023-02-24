# YOLO-POSE based on MMYOLO

Thanks 深度眸.
# train
./tools/train.py configs/yoloxkpt_s_8xb32-300e_coco.py 8

# vis pipeline
python tools/analysis_tools/browse_dataset.py configs/yoloxkpt_s_8xb32-300e_coco.py -n 20 --out-dir vis/pipeline/train -m pipeline -p train

# test
./tools/test.py configs/yoloxkpt_s_8xb32-300e_coco.py xxxx.pth

upload later.

# export to onnx
python ./projects/easydeploy/tools/export.py configs/yoloxkpt_s_8xb32-300e_coco.py xxxx.pth --simplify --model-only

# convert yolox to mmyolo
python tools/model_converters/yolox_to_mmyolo.py --src xxx.pth --dst xxx_2.pth

# experiments

model | COCO kpt mAP
--- | ---
s | 60.8
