# Copyright (c) OpenMMLab. All rights reserved.
from projects.easydeploy.model import ORTWrapper, TRTWrapper  # isort:skip
import os
import random
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg
from mmengine.config import Config, ConfigDict
from mmengine.utils import ProgressBar, path

from mmyolo.utils import register_all_modules
from mmyolo.utils.misc import get_file_list

# if you are using a custom dataset and you did not add class names
# into your model config file, feel free to modify this part if necessary.
CLASS_NAMES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, supports image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    args = parser.parse_args()
    return args


def preprocess(config):
    data_preprocess = config.get('model', {}).get('data_preprocessor', {})
    mean = data_preprocess.get('mean', [0., 0., 0.])
    std = data_preprocess.get('std', [1., 1., 1.])
    mean = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)

    class PreProcess(torch.nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x[None].float()
            x -= mean.to(x.device)
            x /= std.to(x.device)
            return x

    return PreProcess().eval()


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    register_all_modules()

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(1000)]

    # build the model from a config file and a checkpoint file
    if args.checkpoint.endswith('.onnx'):
        model = ORTWrapper(args.checkpoint, args.device)
    elif args.checkpoint.endswith('.engine') or args.checkpoint.endswith(
            '.plan'):
        model = TRTWrapper(args.checkpoint, args.device)
    else:
        raise NotImplementedError

    model.to(args.device)

    cfg = Config.fromfile(args.config)
    class_names = CLASS_NAMES if cfg.get('class_name') is None else cfg.get(
        'class_name')

    test_pipeline = get_test_pipeline_cfg(cfg)
    test_pipeline[0] = ConfigDict({'type': 'mmdet.LoadImageFromNDArray'})
    test_pipeline = Compose(test_pipeline)

    pre_pipeline = preprocess(cfg)

    if not args.show:
        path.mkdir_or_exist(args.out_dir)

    # get file list
    files, source_type = get_file_list(args.img)

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for i, file in enumerate(files):
        bgr = mmcv.imread(file)
        rgb = mmcv.imconvert(bgr, 'bgr', 'rgb')
        data, samples = test_pipeline(dict(img=rgb, img_id=i)).values()
        pad_param = samples.get('pad_param',
                                np.array([0, 0, 0, 0], dtype=np.float32))
        h, w = samples.get('ori_shape', rgb.shape[:2])
        pad_param = torch.asarray(
            [pad_param[2], pad_param[0], pad_param[2], pad_param[0]],
            device=args.device)
        scale_factor = samples.get('scale_factor', [1., 1])
        scale_factor = torch.asarray(scale_factor * 2, device=args.device)
        data = pre_pipeline(data).to(args.device)

        result = model(data)
        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        # Get candidate predict info by num_dets
        num_dets, bboxes, scores, labels = result
        scores = scores[0, :num_dets]
        bboxes = bboxes[0, :num_dets]
        labels = labels[0, :num_dets]
        bboxes -= pad_param
        bboxes /= scale_factor

        bboxes[:, 0::2].clamp_(0, w)
        bboxes[:, 1::2].clamp_(0, h)
        bboxes = bboxes.round().int()

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.tolist()
            color = colors[label]

            label_name = class_names[label]
            name = f'{label_name}: {score:.2f}'

            cv2.rectangle(bgr, bbox[:2], bbox[2:], color, 2)
            cv2.putText(
                bgr,
                name, (bbox[0], bbox[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, [225, 255, 255],
                thickness=2)

        if args.show:
            mmcv.imshow(bgr, 'result', 0)
        else:
            mmcv.imwrite(bgr, out_file)
        progress_bar.update()


if __name__ == '__main__':
    main()
