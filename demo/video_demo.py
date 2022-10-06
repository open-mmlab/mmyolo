# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import mmcv
from mmdet.apis import inference_detector, init_detector

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='MMYOLO video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args


def main():
    # register all modules in mmdet into the registries
    register_all_modules()

    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')

    model = init_detector(args.config, args.checkpoint, device=args.device)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in video_reader:
        result = inference_detector(model, frame)
        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        else:
            visualizer.add_datasample(
                'result',
                frame,
                data_sample=result,
                draw_gt=False,
                show=args.show,
                wait_time=0,
                out_file=None,
                pred_score_thr=args.score_thr)
        if args.out:
            video_writer.write(visualizer.get_image())
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
