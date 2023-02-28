# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import sys
from typing import Tuple

import cv2
import mmcv
import numpy as np
from mmdet.models.utils import mask2ndarray
from mmdet.structures.bbox import BaseBoxes
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar
from mmengine.visualization import Visualizer

from mmyolo.registry import DATASETS, VISUALIZERS


# TODO: Support for printing the change in key of results
def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--phase',
        '-p',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--mode',
        '-m',
        default='transformed',
        type=str,
        choices=['original', 'transformed', 'pipeline'],
        help='display mode; display original pictures or '
        'transformed pictures or comparison pictures. "original" '
        'means show images load from disk; "transformed" means '
        'to show images after transformed; "pipeline" means show all '
        'the intermediate images. Defaults to "transformed".')
    parser.add_argument(
        '--out-dir',
        default='output',
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-number',
        '-n',
        type=int,
        default=sys.maxsize,
        help='number of images selected to visualize, '
        'must bigger than 0. if the number is bigger than length '
        'of dataset, show all the images in dataset; '
        'default "sys.maxsize", show all images in dataset')
    parser.add_argument(
        '--show-interval',
        '-i',
        type=float,
        default=3,
        help='the interval of show (s)')
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


def _get_adaptive_scale(img_shape: Tuple[int, int],
                        min_scale: float = 0.3,
                        max_scale: float = 3.0) -> float:
    """Get adaptive scale according to image shape.

    The target scale depends on the the short edge length of the image. If the
    short edge length equals 224, the output is 1.0. And output linear
    scales according the short edge length. You can also specify the minimum
    scale and the maximum scale to limit the linear scale.

    Args:
        img_shape (Tuple[int, int]): The shape of the canvas image.
        min_scale (int): The minimum scale. Defaults to 0.3.
        max_scale (int): The maximum scale. Defaults to 3.0.
    Returns:
        int: The adaptive scale.
    """
    short_edge_length = min(img_shape)
    scale = short_edge_length / 224.
    return min(max(scale, min_scale), max_scale)


def make_grid(imgs, names):
    """Concat list of pictures into a single big picture, align height here."""
    visualizer = Visualizer.get_current_instance()
    ori_shapes = [img.shape[:2] for img in imgs]
    max_height = int(max(img.shape[0] for img in imgs) * 1.1)
    min_width = min(img.shape[1] for img in imgs)
    horizontal_gap = min_width // 10
    img_scale = _get_adaptive_scale((max_height, min_width))

    texts = []
    text_positions = []
    start_x = 0
    for i, img in enumerate(imgs):
        pad_height = (max_height - img.shape[0]) // 2
        pad_width = horizontal_gap // 2
        # make border
        imgs[i] = cv2.copyMakeBorder(
            img,
            pad_height,
            max_height - img.shape[0] - pad_height + int(img_scale * 30 * 2),
            pad_width,
            pad_width,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255))
        texts.append(f'{"execution: "}{i}\n{names[i]}\n{ori_shapes[i]}')
        text_positions.append(
            [start_x + img.shape[1] // 2 + pad_width, max_height])
        start_x += img.shape[1] + horizontal_gap

    display_img = np.concatenate(imgs, axis=1)
    visualizer.set_image(display_img)
    img_scale = _get_adaptive_scale(display_img.shape[:2])
    visualizer.draw_texts(
        texts,
        positions=np.array(text_positions),
        font_sizes=img_scale * 7,
        colors='black',
        horizontal_alignments='center',
        font_families='monospace')
    return visualizer.get_image()


class InspectCompose(Compose):
    """Compose multiple transforms sequentially.

    And record "img" field of all results in one list.
    """

    def __init__(self, transforms, intermediate_imgs):
        super().__init__(transforms=transforms)
        self.intermediate_imgs = intermediate_imgs

    def __call__(self, data):
        if 'img' in data:
            self.intermediate_imgs.append({
                'name': 'original',
                'img': data['img'].copy()
            })
        self.ptransforms = [
            self.transforms[i] for i in range(len(self.transforms) - 1)
        ]
        for t in self.ptransforms:
            data = t(data)
            # Keep the same meta_keys in the PackDetInputs
            self.transforms[-1].meta_keys = [key for key in data]
            data_sample = self.transforms[-1](data)
            if data is None:
                return None
            if 'img' in data:
                self.intermediate_imgs.append({
                    'name':
                    t.__class__.__name__,
                    'dataset_sample':
                    data_sample['data_samples']
                })
        return data


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmyolo'))

    dataset_cfg = cfg.get(args.phase + '_dataloader').get('dataset')
    dataset = DATASETS.build(dataset_cfg)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    intermediate_imgs = []

    if not hasattr(dataset, 'pipeline'):
        # for dataset_wrapper
        dataset = dataset.dataset

    # TODO: The dataset wrapper occasion is not considered here
    dataset.pipeline = InspectCompose(dataset.pipeline.transforms,
                                      intermediate_imgs)

    # init visualization image number
    assert args.show_number > 0
    display_number = min(args.show_number, len(dataset))

    progress_bar = ProgressBar(display_number)
    for i, item in zip(range(display_number), dataset):
        image_i = []
        result_i = [result['dataset_sample'] for result in intermediate_imgs]
        for k, datasample in enumerate(result_i):
            image = datasample.img
            gt_instances = datasample.gt_instances
            image = image[..., [2, 1, 0]]  # bgr to rgb
            gt_bboxes = gt_instances.get('bboxes', None)
            if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
                gt_instances.bboxes = gt_bboxes.tensor
            gt_masks = gt_instances.get('masks', None)
            if gt_masks is not None:
                masks = mask2ndarray(gt_masks)
                gt_instances.masks = masks.astype(bool)
                datasample.gt_instances = gt_instances
            # get filename from dataset or just use index as filename
            visualizer.add_datasample(
                'result',
                image,
                datasample,
                draw_pred=False,
                draw_gt=True,
                show=False)
            image_show = visualizer.get_image()
            image_i.append(image_show)

        if args.mode == 'original':
            image = image_i[0]
        elif args.mode == 'transformed':
            image = image_i[-1]
        else:
            image = make_grid([result for result in image_i],
                              [result['name'] for result in intermediate_imgs])

        if hasattr(datasample, 'img_path'):
            filename = osp.basename(datasample.img_path)
        else:
            # some dataset have not image path
            filename = f'{i}.jpg'
        out_file = osp.join(args.out_dir,
                            filename) if args.out_dir is not None else None

        if out_file is not None:
            mmcv.imwrite(image[..., ::-1], out_file)

        if not args.not_show:
            visualizer.show(
                image, win_name=filename, wait_time=args.show_interval)

        intermediate_imgs.clear()
        progress_bar.update()


if __name__ == '__main__':
    main()
