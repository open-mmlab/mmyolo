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
from mmyolo.utils import register_all_modules

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--deploy_cfg', type=str, default=None, help='deploy config path')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def switch_deploy(args):
    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import get_input_shape, load_config

    deploy_cfg, model_cfg = load_config(args.deploy_cfg, args.config)
    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)
    model = task_processor.build_backend_model([args.checkpoint])
    input_shape = get_input_shape(deploy_cfg)
    visualizer = task_processor.get_visualizer(
        name='result', save_dir=args.out_dir)
    return model, task_processor, input_shape, visualizer


def switch_pytorch(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    return model, visualizer


def main(args):
    # register all modules in mmdet into the registries
    register_all_modules()

    if args.deploy_cfg is None:
        model, visualizer = switch_pytorch(args)
    else:
        model, task_processor, input_shape, visualizer = switch_deploy(args)

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
        img = mmcv.imread(file)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        if is_dir:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        if args.deploy_cfg is None:
            result = inference_detector(model, file)
        else:
            data, _ = task_processor.create_input(file, input_shape)
            result = model.test_step(data)[0]

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
