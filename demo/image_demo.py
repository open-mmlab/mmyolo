# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.logging import print_log
from mmengine.utils import ProgressBar

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules, switch_to_deploy
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.utils.misc import get_file_list


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
    parser.add_argument(
        '--to-labelme',
        action='store_true',
        help='Output labelme style label file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.to_labelme and args.show:
        raise RuntimeError('`--to-labelme` or `--show` only '
                           'can choose one at the same time.')

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    if args.deploy:
        switch_to_deploy(model)

    if not os.path.exists(args.out_dir) and not args.show:
        os.mkdir(args.out_dir)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # get file list
    files, source_type = get_file_list(args.img)

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        result = inference_detector(model, file)

        img = mmcv.imread(file)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        progress_bar.update()
        if args.to_labelme:
            # save result to labelme files
            out_file = out_file.replace(
                os.path.splitext(out_file)[-1], '.json')
            to_label_format = LabelmeFormat()
            to_label_format(
                result,
                out_file,
                pred_score_thr=args.score_thr,
                model_classes=model.dataset_meta.get('CLASSES'))
            continue

        visualizer.add_datasample(
            filename,
            img,
            data_sample=result,
            draw_gt=False,
            show=args.show,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=args.score_thr)

    if not args.show and not args.to_labelme:
        print_log(
            f'\nResults have been saved at {os.path.abspath(args.out_dir)}')

    elif args.to_labelme:
        print_log('\nLabelme format label files '
                  f'had all been saved in {args.out_dir}')


if __name__ == '__main__':
    main()
