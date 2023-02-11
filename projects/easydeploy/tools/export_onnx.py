import argparse
import os
import warnings
from io import BytesIO

import onnx
import torch
from mmdet.apis import init_detector
from mmengine.config import ConfigDict
from mmengine.utils.path import mkdir_or_exist

from mmyolo.utils import register_all_modules
from projects.easydeploy.model import DeployModel

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
warnings.filterwarnings(action='ignore', category=torch.jit.ScriptWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=ResourceWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--model-only', action='store_true', help='Export model only')
    parser.add_argument(
        '--work-dir', default='./work_dir', help='Path to save export model')
    parser.add_argument(
        '--img-size',
        nargs='+',
        type=int,
        default=[640, 640],
        help='Image size of height and width')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify onnx model by onnx-sim')
    parser.add_argument(
        '--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument(
        '--backend', type=int, default=1, help='Backend for export onnx')
    parser.add_argument(
        '--pre-topk',
        type=int,
        default=1000,
        help='Postprocess pre topk bboxes feed into NMS')
    parser.add_argument(
        '--keep-topk',
        type=int,
        default=100,
        help='Postprocess keep topk bboxes out of NMS')
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.65,
        help='IoU threshold for NMS')
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.25,
        help='Score threshold for NMS')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1
    return args


def build_model_from_cfg(config_path, checkpoint_path, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model


def main():
    args = parse_args()
    register_all_modules()

    mkdir_or_exist(args.work_dir)

    if args.model_only:
        postprocess_cfg = None
        output_names = None
    else:
        postprocess_cfg = ConfigDict(
            pre_top_k=args.pre_topk,
            keep_top_k=args.keep_topk,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            backend=args.backend)
        output_names = ['num_dets', 'boxes', 'scores', 'labels']
    baseModel = build_model_from_cfg(args.config, args.checkpoint, args.device)

    deploy_model = DeployModel(
        baseModel=baseModel, postprocess_cfg=postprocess_cfg)
    deploy_model.eval()

    fake_input = torch.randn(args.batch_size, 3,
                             *args.img_size).to(args.device)
    # dry run
    deploy_model(fake_input)

    save_onnx_path = os.path.join(args.work_dir, 'end2end.onnx')
    # export onnx
    with BytesIO() as f:
        torch.onnx.export(
            deploy_model,
            fake_input,
            f,
            input_names=['images'],
            output_names=output_names,
            opset_version=args.opset)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)

        # Fix tensorrt onnx output shape, just for view
        if args.backend in (2, 3):
            shapes = [
                args.batch_size, 1, args.batch_size, args.keep_topk, 4,
                args.batch_size, args.keep_topk, args.batch_size,
                args.keep_topk
            ]
            for i in onnx_model.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))
    if args.simplify:
        try:
            import onnxsim
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplify failure: {e}')
    onnx.save(onnx_model, save_onnx_path)
    print(f'ONNX export success, save into {save_onnx_path}')


if __name__ == '__main__':
    main()
