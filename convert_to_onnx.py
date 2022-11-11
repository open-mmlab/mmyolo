import torch
from mmdet.apis import init_detector
from mmengine.config import ConfigDict

from mmyolo.easydeploy import DeployModel
from mmyolo.utils import register_all_modules, switch_to_deploy


def build_model_from_cfg(config_path, checkpoint_path, device, switch=False):
    model = init_detector(config_path, checkpoint_path, device=device)
    if switch:
        switch_to_deploy(model)
    model.eval()
    return model


if __name__ == '__main__':
    config = './configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'
    checkpoint = './yolov5s.pth'
    device = torch.device('cuda:0')
    register_all_modules()
    baseModel = build_model_from_cfg(config, checkpoint, device)
    inp = torch.randn(1, 3, 640, 640).to(device)
    postprocess_cfg = ConfigDict(
        pre_top_k=1000,
        keep_top_k=100,
        iou_threshold=0.65,
        score_threshold=0.25,
    )
    deploy_model = DeployModel(
        baseModel=baseModel, postprocess_cfg=postprocess_cfg)
    deploy_model.eval()
    # dry run
    deploy_model(inp)
    torch.onnx.export(
        deploy_model,
        inp,
        'tmp.onnx',
        input_names=['images'],
        output_names=['num_det', 'det_boxes', 'det_scores', 'det_classes'],
        opset_version=11)
