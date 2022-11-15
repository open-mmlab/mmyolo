# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmengine.logging import print_log
from mmengine.utils import ProgressBar
from sahi.slicing import slice_image

from demo.large_image_demo_utils import merge_results_by_nms
from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules, switch_to_deploy
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
        '--patch_size', type=int, default=1024, help='The size of patches')
    parser.add_argument(
        '--patch_overlap_ratio',
        type=int,
        default=0.25,
        help='Ratio of overlap between two patches')
    parser.add_argument(
        '--merge_iou_thr',
        type=float,
        default=0.1,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size, must greater than or equal to 1')
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
        # read image
        if not isinstance(args.img, np.ndarray):
            img = mmcv.imread(args.img)

        # arrange slices
        height, width = img.shape[:2]
        sliced_images = slice_image(
            img,
            slice_height=args.patch_size,
            slice_width=args.patch_size,
            auto_slice_resolution=False,
            overlap_height_ratio=args.patch_overlap_ratio,
            overlap_width_ratio=args.patch_overlap_ratio,
        )

        # perform sliced inference
        results = []
        start = 0
        while True:
            # prepare batch slices
            end = min(start + args.batch_size, len(sliced_images))
            images = []
            for sliced_image in sliced_images[start:end]:
                images.append(sliced_image['image'])

            # forward the model
            results.extend(inference_detector(model, images))

            if end >= len(sliced_images):
                break
            start += args.batch_size

        image_result = merge_results_by_nms(
            results,
            sliced_images.starting_pixels,
            full_shape=(height, width),
            nms_cfg={
                'type': 'nms',
                'iou_thr': 0.25
            })

        img = mmcv.imconvert(img, 'bgr', 'rgb')

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        visualizer.add_datasample(
            os.path.basename(out_file),
            img,
            data_sample=image_result,
            draw_gt=False,
            show=args.show,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=args.score_thr,
        )
        progress_bar.update()

    if not args.show:
        print_log(
            f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
