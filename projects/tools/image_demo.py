# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

import mmcv
import torch
from mmcv.transforms import Compose
from mmengine import Config
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from mmengine.utils import ProgressBar

from mmyolo.registry import DATASETS, VISUALIZERS
from mmyolo.utils import register_all_modules
from mmyolo.utils.misc import get_file_list
from projects.easydeploy import BackendWrapper


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--backend', type=int, default=1, help='Backend for export onnx')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    register_all_modules()
    config = Config.fromfile(args.config)
    model = BackendWrapper(args.checkpoint, args.device)
    pipeline = Compose(config.get('test_pipeline'))
    visualizer = VISUALIZERS.build(config.get('visualizer'))
    dataset = DATASETS.build(config.get('test_dataloader', {}).get('dataset'))
    visualizer.dataset_meta = getattr(dataset, 'METAINFO')

    if not os.path.exists(args.out_dir) and not args.show:
        os.mkdir(args.out_dir)

    # get file list
    files, source_type = get_file_list(args.img)

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        inputs, data_samples = pipeline(dict(img_path=file,
                                             img_id=None)).values()
        instance_data = InstanceData(metainfo=data_samples.metainfo)
        tensor = inputs[None]
        if args.backend == 1:
            tensor = tensor.float().contiguous().cpu().numpy()
            num_det, det_bboxes, det_scores, det_labels = (
                torch.from_numpy(data).to(model.device)
                for data in model(tensor))
        elif args.backend in (2, 3):
            tensor = tensor.float().to(model.device).contiguous()
            num_det, det_bboxes, det_scores, det_labels = model(tensor)
        else:
            raise NotImplementedError

        num_det = num_det.item()
        det_bboxes, det_scores, det_labels = det_bboxes[
            0, :num_det], det_scores[0, :num_det], det_labels[0, :num_det]
        instance_data.det_labels = det_labels
        instance_data.det_scores = det_scores
        instance_data.bboxes = det_bboxes
        data_samples._pred_instances = instance_data

        img = mmcv.imread(file, channel_order='rgb')

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        visualizer.add_datasample(
            out_file,
            img,
            data_sample=data_samples,
            draw_gt=False,
            show=args.show,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=args.score_thr)
        progress_bar.update()

    if not args.show:
        print_log(
            f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
