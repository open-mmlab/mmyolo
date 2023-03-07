import argparse
import os
import warnings

import torch
from mmdet.apis import init_detector
from mmengine.logging import print_log
from mmengine.utils.path import mkdir_or_exist

from projects.easydeploy.model import DeployModel, MMYOLOBackend

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

    baseModel = build_model_from_cfg(args.config, args.checkpoint, args.device)

    deploy_model = DeployModel(
        baseModel=baseModel,
        backend=MMYOLOBackend.TORCHSCRIPT,
        postprocess_cfg=None)
    deploy_model.eval()

    fake_input = torch.randn(args.batch_size, 3,
                             *args.img_size).to(args.device)
    # dry run
    deploy_model(fake_input)

    save_torchscript_path = os.path.join(
        args.work_dir,
        os.path.basename(args.checkpoint).replace('pth', 'torchscript'))
    mod = torch.jit.trace(deploy_model, fake_input)
    mod.save(save_torchscript_path)
    print_log(f'TORCHSCRIPT export success, save into {save_torchscript_path}')


if __name__ == '__main__':
    main()
