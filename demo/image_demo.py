# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import urllib
from argparse import ArgumentParser

import mmcv
import torch
from mmdet.apis import inference_detector, init_detector
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, scandir

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules, switch_to_deploy

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


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


def main(args):
    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    if args.deploy:
        switch_to_deploy(model)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    is_dir = os.path.isdir(args.img)
    is_url = args.img.startswith(('http:/', 'https:/'))
    is_file = os.path.splitext(args.img)[-1] in (IMG_EXTENSIONS)

    files = []
    if is_dir:
        # when input source is dir
        for file in scandir(args.img, IMG_EXTENSIONS, recursive=True):
            files.append(os.path.join(args.img, file))
    elif is_url:
        # when input source is url
        filename = os.path.basename(
            urllib.parse.unquote(args.img).split('?')[0])
        torch.hub.download_url_to_file(args.img, filename)
        files = [os.path.join(os.getcwd(), filename)]
    elif is_file:
        # when input source is single image
        files = [args.img]
    else:
        print_log(
            'Cannot find image file.', logger='current', level=logging.WARNING)

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        result = inference_detector(model, file)
        img = mmcv.imread(file)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        if is_dir:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)
        visualizer.add_datasample(
            filename,
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
    args = parse_args()
    main(args)
