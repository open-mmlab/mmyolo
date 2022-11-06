# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import urllib
from typing import Sequence

import mmcv
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmengine import Config, DictAction
from mmengine.utils import ProgressBar, scandir

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


# TODO: Refine
def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature map')
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-dir', default=None, help='Path to output file')
    parser.add_argument(
        '--show', action='store_true', help='Show the feature map results')
    parser.add_argument(
        '--target-layers',
        default=['backbone'],
        nargs='+',
        type=str,
        help='The target layers to get feature map, if not set, the tool will '
        'specify the backbone')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--channel-reduction',
        default='select_max',
        help='Reduce multiple channels to a single channel')
    parser.add_argument(
        '--topk',
        type=int,
        default=4,
        help='Select topk channel to show by the sum of each channel')
    parser.add_argument(
        '--arrangement',
        nargs='+',
        type=int,
        default=[2, 2],
        help='The arrangement of featmap when channel_reduction is '
        'not None and topk > 0')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


class ActivationsWrapper:

    def __init__(self, model, target_layers):
        self.model = model
        self.activations = []
        self.handles = []
        self.image = None
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def __call__(self, img_path):
        self.activations = []
        results = inference_detector(self.model, img_path)
        return results, self.activations

    def release(self):
        for handle in self.handles:
            handle.remove()


def auto_arrange_imgs(imgs):
    len_img = len(imgs)
    col = 2
    if len_img <= col:
        imgs = np.concatenate(imgs, axis=1)
    else:
        row = round(len_img / col)
        fill_img_list = [np.ones(imgs[0].shape, dtype=np.uint8) * 255] * (
            row * col - len_img)
        imgs.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(row):
            start = col * i
            end = col * (i + 1)
            merge_col = np.hstack(imgs[start:end])
            merge_imgs_col.append(merge_col)

        imgs = np.vstack(merge_imgs_col)
    return imgs


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    register_all_modules()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    channel_reduction = args.channel_reduction
    if channel_reduction == 'None':
        channel_reduction = None
    assert len(args.arrangement) == 2

    model = init_detector(args.config, args.checkpoint, device=args.device)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if args.preview_model:
        print(model)
        print('\n This flag is only show model, if you want to continue, '
              'please remove `--preview-model` to get the feature map.')
        return

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(eval(f'model.{target_layer}'))
        except Exception as e:
            print(model)
            raise RuntimeError('layer does not exist', e)

    activations_wrapper = ActivationsWrapper(model, target_layers)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    is_dir = os.path.isdir(args.img)
    is_url = args.img.startswith(('http:/', 'https:/'))
    is_file = os.path.splitext(args.img)[-1] in IMG_EXTENSIONS

    image_list = []
    if is_dir:
        # when input source is dir
        for file in scandir(args.img, IMG_EXTENSIONS, recursive=True):
            image_list.append(os.path.join(args.img, file))
    elif is_url:
        # when input source is url
        filename = os.path.basename(
            urllib.parse.unquote(args.img).split('?')[0])
        torch.hub.download_url_to_file(args.img, filename)
        image_list = [os.path.join(os.getcwd(), filename)]
    elif is_file:
        # when input source is single image
        image_list = [args.img]
    else:
        print('Cannot find image file.')

    progress_bar = ProgressBar(len(image_list))
    for image_path in image_list:
        result, featmaps = activations_wrapper(image_path)
        if not isinstance(featmaps, Sequence):
            featmaps = [featmaps]

        flatten_featmaps = []
        for featmap in featmaps:
            if isinstance(featmap, Sequence):
                flatten_featmaps.extend(featmap)
            else:
                flatten_featmaps.append(featmap)

        # show the results
        img = mmcv.imread(image_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        shown_imgs = []
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=None,
            pred_score_thr=args.score_thr)
        drawn_img = visualizer.get_image()

        for featmap in flatten_featmaps:
            shown_img = visualizer.draw_featmap(
                featmap[0],
                drawn_img,
                channel_reduction=channel_reduction,
                topk=args.topk,
                arrangement=args.arrangement)
            shown_imgs.append(shown_img)

        # Add original image
        shown_imgs.append(img)
        shown_imgs = auto_arrange_imgs(shown_imgs)

        progress_bar.update()
        if args.out_dir is not None:
            mmcv.imwrite(
                shown_imgs[..., ::-1],
                os.path.join(args.out_dir, os.path.basename(image_path)))
        else:
            visualizer.show(shown_imgs)

    print('All done!')


# Please refer to the usage tutorial:
# https://github.com/open-mmlab/mmyolo/blob/main/docs/zh_cn/user_guides/visualization.md # noqa
if __name__ == '__main__':
    main()
