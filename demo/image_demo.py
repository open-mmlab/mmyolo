# Copyright (c) OpenMMLab. All rights reserved.
import json
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
        '--save-json-type',
        default='',
        choices=['cwh', 'tl-wh', 'tl-br'],
        help='Save predict info to json with type in ("cwh", "tl-wh", "tl-br"). '
        '`cwh`: center xy and bbox wh, '
        '`tl-wh`: top-left xy and bbox wh, '
        '`tl-br`: top-left xy and bottom-right xy.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    # get argument
    args = parse_args()

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    is_dir = os.path.isdir(args.img)
    is_url = args.img.startswith(('http:/', 'https:/'))
    is_file = os.path.splitext(args.img)[-1] in IMG_EXTENSIONS

    # save predict info to predict.json
    save_pred_json = args.save_json_type != ''
    pred_json_path = ''
    pred_json = []
    if save_pred_json and not args.show:
        pred_json_path = f'{args.out_dir}/predict.json'
        print_log(f'Predict info json files will be saved in {pred_json_path}')

    files = []
    if is_dir and os.path.exists(args.img):
        # when input source is dir
        for file in scandir(args.img, IMG_EXTENSIONS, recursive=True):
            files.append(os.path.join(args.img, file))
    elif is_url:
        # when input source is url
        filename = os.path.basename(
            urllib.parse.unquote(args.img).split('?')[0])
        torch.hub.download_url_to_file(args.img, filename)
        files = [os.path.join(os.getcwd(), filename)]
    elif is_file and os.path.exists(args.img):
        # when input source is single image
        files = [args.img]
    else:
        raise FileNotFoundError('Cannot find image file.')

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

        if save_pred_json:
            for idx in range(len(result.pred_instances['labels'])):
                # save json as
                # {"image_id": 'xxx', "category_id": 66, "bbox": [66.551, 23.462, 123.564, 59.336], "score": 0.97557}
                if not result.pred_instances['scores'][idx] >= args.score_thr:
                    continue

                bbox = result.pred_instances['bboxes'][idx].tolist()
                if args.save_json_type in ['cwh', 'tl-wh']:
                    bbox_save = [0, 0, 0, 0]
                    if args.save_json_type == 'tl-wh':
                        bbox_save[0] = bbox[0]  # left top x
                        bbox_save[1] = bbox[1]  # left top y
                    else:
                        bbox_save[0] = (bbox[0] + bbox[2]) / 2  # center x
                        bbox_save[1] = (bbox[1] + bbox[3]) / 2  # center y
                    bbox_save[2] = bbox[2] - bbox[0]  # bbox width
                    bbox_save[3] = bbox[3] - bbox[1]  # bbox height
                else:
                    # top-left xy and bottom-right xy
                    bbox_save = result.pred_instances['bboxes'][idx].tolist()

                pred_json.append({
                    'image_id':
                    os.path.splitext(filename)[0],
                    'category_id':
                    int(result.pred_instances['labels'][idx]),
                    'bbox': [round(x, 3) for x in bbox_save],
                    'score':
                    round(result.pred_instances['scores'][idx].tolist(), 5)
                })

        progress_bar.update()

    if save_pred_json:
        print_log(f'Saving predict info files to {pred_json_path}')
        with open(pred_json_path, 'w') as f:
            json.dump(pred_json, f)

    if not args.show:
        print_log(
            f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
