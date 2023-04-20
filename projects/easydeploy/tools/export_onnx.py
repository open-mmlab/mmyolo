import argparse
import os
import sys
import warnings
from io import BytesIO
from pathlib import Path

import onnx
import torch
from mmdet.apis import init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils.path import mkdir_or_exist
from torch.nn.parameter import Parameter

# Add MMYOLO ROOT to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from projects.easydeploy.model import DeployModel, MMYOLOBackend  # noqa E402

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
warnings.filterwarnings(action='ignore', category=torch.jit.ScriptWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=ResourceWarning)


def preprocess(config: Config, model: torch.nn.Module):
    data_preprocess = config.get('model', {}).get('data_preprocessor', {})
    mean = data_preprocess.get('mean', [0., 0., 0.])
    std = data_preprocess.get('std', [1., 1., 1.])
    mean_value = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
    std_value = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)

    class PreProcess(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.mean_value = Parameter(mean_value, requires_grad=False)
            self.std_value = Parameter(std_value, requires_grad=False)
            self.core_model = model

        def forward(self, x: torch.Tensor):
            assert x.ndim == 4
            x = x.float()
            y = (x - self.mean_value) / self.std_value
            y = self.core_model(y)
            return y

    return PreProcess().eval()


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
        '--backend',
        type=str,
        default='onnxruntime',
        help='Backend for export onnx')
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
    mkdir_or_exist(args.work_dir)
    backend = MMYOLOBackend(args.backend.lower())
    if backend in (MMYOLOBackend.ONNXRUNTIME, MMYOLOBackend.OPENVINO,
                   MMYOLOBackend.TENSORRT8, MMYOLOBackend.TENSORRT7):
        if not args.model_only:
            print_log('Export ONNX with bbox decoder and NMS ...')
    else:
        args.model_only = True
        print_log(f'Can not export postprocess for {args.backend.lower()}.\n'
                  f'Set "args.model_only=True" default.')
    if args.model_only:
        postprocess_cfg = None
        output_names = None
    else:
        postprocess_cfg = ConfigDict(
            pre_top_k=args.pre_topk,
            keep_top_k=args.keep_topk,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold)
        output_names = ['num_dets', 'boxes', 'scores', 'labels']
    baseModel = build_model_from_cfg(args.config, args.checkpoint, args.device)

    deploy_model = DeployModel(
        baseModel=baseModel, backend=backend, postprocess_cfg=postprocess_cfg)
    deploy_model.eval()

    # embed the preprocess into the model
    cfg = Config.fromfile(args.config)
    deploy_model = preprocess(cfg, deploy_model)
    deploy_model.eval()

    fake_input = torch.randn(args.batch_size, 3,
                             *args.img_size).type(torch.uint8).to(args.device)
    # dry run
    deploy_model(fake_input)

    save_onnx_path = os.path.join(
        args.work_dir,
        os.path.basename(args.checkpoint).replace('pth', 'onnx'))
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
        if not args.model_only and backend in (MMYOLOBackend.TENSORRT8,
                                               MMYOLOBackend.TENSORRT7):
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
            print_log(f'Simplify failure: {e}')
    onnx.save(onnx_model, save_onnx_path)
    print_log(f'ONNX export success, save into {save_onnx_path}')


if __name__ == '__main__':
    main()
