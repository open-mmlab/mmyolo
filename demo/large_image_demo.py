# Copyright (c) OpenMMLab. All rights reserved.
"""Perform MMYOLO inference on large images (as satellite imagery) as:

```shell
wget -P checkpoint https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth   syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth  # noqa: E501, E261.

python demo/large_image_demo.py \
    demo/demo.jpg \
    configs/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco.py \
    checkpoint/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth \
```
"""

import os
from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.logging import print_log
from mmengine.utils import ProgressBar
from sahi.slicing import slice_image

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules, switch_to_deploy
from mmyolo.utils.large_image import merge_results_by_nms
from mmyolo.utils.misc import get_file_list


def parse_args():
    parser = ArgumentParser(
        description='Perform MMYOLO inference on large images.')
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
        '--patch-size', type=int, default=640, help='The size of patches')
    parser.add_argument(
        '--patch-overlap-ratio',
        type=int,
        default=0.25,
        help='Ratio of overlap between two patches')
    parser.add_argument(
        '--merge-iou-thr',
        type=float,
        default=0.25,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--batch-size',
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
        img = mmcv.imread(file)

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
                'iou_thr': args.merge_iou_thr
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
