# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector, init_detector

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif',\
     'tiff', 'webp', 'pfm'


class LoadFiles:

    def __init__(self, path):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(os.path.abspath(p))
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(
                    sorted(
                        glob.glob(os.path.join(p, '**/*.*'),
                                  recursive=True)))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')
        self.files = [
            x for x in files if x.split('.')[-1].lower() in IMG_FORMATS
        ]
        self.num_files = len(self.files)
        self.count = 0
        assert self.num_files > 0, f'No images found in {p}.'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.num_files:
            raise StopIteration
        path = self.files[self.count]
        self.count += 1
        return path

    def __len__(self):
        return self.num_files


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-path', default='./', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    files = LoadFiles(args.img)
    for file in files:
        result = inference_detector(model, file)
        img = mmcv.imread(file)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        _, file_name = os.path.split(file)
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=args.show,
            wait_time=0,
            out_file=os.path.join(args.out_path, file_name),
            pred_score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
