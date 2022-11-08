# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector
from mmengine.logging import print_log
from mmengine.utils import ProgressBar

from demo.utils import get_file_list, get_image_and_out_file_path
from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules, switch_to_deploy


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    if args.deploy:
        switch_to_deploy(model)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # get file list
    files, source_type = get_file_list(args.img)

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        result = inference_detector(model, file)

        # get original image and out save path if it is needed.
        img, out_file = get_image_and_out_file_path(file, args.img, source_type['is_dir'],
                                                    args.out_dir, args.show)

        visualizer.add_datasample(
            os.path.basename(out_file),
            img,
            data_sample=result,
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
