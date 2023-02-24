# YOLO-POSE based on MMYOLO

Thanks 深度眸.
# install
```bash
pip install -U openmim
python -m mim install "mmengine>=0.3.1"
python -m mim install "mmcv>=2.0.0rc1,<2.1.0"
python -m mim install "mmdet>=3.0.0rc5,<3.1.0"
python -m mim install "mmpose>=1.0.0b0"
python -m pip install -r requirements/albu.txt
python -m mim install -v -e .
```
# train
```
./tools/train.py configs/yoloxkpt_s_8xb32-300e_coco.py 8
```
# vis pipeline
```
python tools/analysis_tools/browse_dataset.py configs/yoloxkpt_s_8xb32-300e_coco.py -n 20 --out-dir vis/pipeline/train -m pipeline -p train
```
# test
```
./tools/test.py configs/yoloxkpt_s_8xb32-300e_coco.py xxxx.pth
```
model weights will be uploaded later, busy with chatGPT.

# export to onnx
```
python ./projects/easydeploy/tools/export.py configs/yoloxkpt_s_8xb32-300e_coco.py xxxx.pth --simplify --model-only
```
# convert yolox to mmyolo
```
python tools/model_converters/yolox_to_mmyolo.py --src xxx.pth --dst xxx_2.pth
```
# experiments

model | COCO kpt mAP
--- | ---
s | 60.8
