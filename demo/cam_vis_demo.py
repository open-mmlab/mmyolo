# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path
from functools import partial

import cv2
import mmcv
import mmengine
from mmengine import Config, DictAction

from mmyolo.utils import register_all_modules
from mmyolo.utils.cam_utils import (CAMDetectorVisualizer, CAMDetectorWrapper,
                                    DetAblationLayer, DetBoxScoreTarget,
                                    reshape_transform)

try:
    from pytorch_grad_cam import (AblationCAM, EigenCAM, EigenGradCAM, GradCAM,
                                  GradCAMPlusPlus, LayerCAM, XGradCAM)
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      'pytorch_grad_cam package.')

GRAD_FREE_METHOD_MAP = {
    'ablationcam': AblationCAM,
    'eigencam': EigenCAM,
    # 'scorecam': ScoreCAM, # consumes too much memory
}

GRAD_BASED_METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM
}

ALL_SUPPORT_METHODS = list(GRAD_FREE_METHOD_MAP.keys()
                           | GRAD_BASED_METHOD_MAP.keys())


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--method',
        default='eigencam',
        choices=ALL_SUPPORT_METHODS,
        help='Type of method to use, supports '
        f'{", ".join(ALL_SUPPORT_METHODS)}.')
    parser.add_argument(
        '--target-layers',
        default=['backbone.stage4'],
        nargs='+',
        type=str,
        help='The target layers to get CAM, if not set, the tool will '
        'specify the backbone.stage4')
    parser.add_argument('--out-dir', default=None, help='Path to output file')
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
        '--topk',
        type=int,
        default=4,
        help='Select topk predict resutls to show')
    parser.add_argument(
        '--max-shape',
        nargs='+',
        type=int,
        default=-1,
        help='max shapes. Its purpose is to save GPU memory. '
        'The activation map is scaled and then evaluated. '
        'If set to -1, it means no scaling.')
    parser.add_argument(
        '--no-norm-in-bbox',
        action='store_true',
        help='Norm in bbox of cam image')
    # Visualization settings
    parser.add_argument(
        '--channel-reduction',
        default='select_max',
        help='Reduce multiple channels to a single channel')
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
    # Only used by AblationCAM
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='batch of inference of AblationCAM')
    parser.add_argument(
        '--ratio-channels-to-ablate',
        type=int,
        default=0.5,
        help='Making it much faster of AblationCAM. '
        'The parameter controls how many channels should be ablated')

    args = parser.parse_args()
    return args


def init_detector_and_visualizer(args, cfg):
    max_shape = args.max_shape
    if not isinstance(max_shape, list):
        max_shape = [args.max_shape]
    assert len(max_shape) == 1 or len(max_shape) == 2

    model_wrapper = CAMDetectorWrapper(
        cfg, args.checkpoint, args.score_thr, device=args.device)

    if args.preview_model:
        print(model_wrapper.detector)
        print('\n Please remove `--preview-model` to get the CAM.')
        return None, None

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(
                eval(f'model_wrapper.detector.{target_layer}'))
        except Exception as e:
            print(model_wrapper.detector)
            raise RuntimeError('layer does not exist', e)

    ablationcam_extra_params = {
        'batch_size': args.batch_size,
        'ablation_layer': DetAblationLayer(),
        'ratio_channels_to_ablate': args.ratio_channels_to_ablate
    }

    if args.method in GRAD_BASED_METHOD_MAP:
        method_class = GRAD_BASED_METHOD_MAP[args.method]
        is_need_grad = True
        # assert args.no_norm_in_bbox is False, 'If not norm in bbox, the ' \
        #                                       'visualization result ' \
        #                                       'may not be reasonable.'
    else:
        method_class = GRAD_FREE_METHOD_MAP[args.method]
        is_need_grad = False

    cam_detector_visualizer = CAMDetectorVisualizer(
        method_class,
        model_wrapper,
        target_layers,
        reshape_transform=partial(
            reshape_transform, max_shape=max_shape, is_need_grad=is_need_grad),
        is_need_grad=is_need_grad,
        extra_params=ablationcam_extra_params)
    return model_wrapper, cam_detector_visualizer


def main():
    register_all_modules()

    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model_wrapper, cam_detector_visualizer = init_detector_and_visualizer(
        args, cfg)

    images = args.img
    if not isinstance(images, list):
        images = [images]

    for image_path in images:
        image = cv2.imread(image_path)
        model_wrapper.set_input_data(image)

        result = model_wrapper()

        pred_instances = result.pred_instances
        assert len(pred_instances) > 0, 'empty detect bboxes'
        if args.topk > 0:
            pred_instances = pred_instances[:args.topk]

        targets = [DetBoxScoreTarget(pred_instances)]

        if args.method in GRAD_BASED_METHOD_MAP:
            model_wrapper.need_loss(True)
            model_wrapper.set_input_data(image, bboxes=bboxes, labels=labels)
            cam_detector_visualizer.switch_activations_and_grads(model_wrapper)

        grayscale_cam = cam_detector_visualizer(image, targets=targets)

        pred_instances = pred_instances.numpy()
        image_with_bounding_boxes = cam_detector_visualizer.show_cam(
            image, pred_instances.bboxes, pred_instances.labels, grayscale_cam,
            not args.no_norm_in_bbox)

        if args.out_dir:
            mmengine.mkdir_or_exist(args.out_dir)
            out_file = os.path.join(args.out_dir, os.path.basename(image_path))
            mmcv.imwrite(image_with_bounding_boxes, out_file)
        else:
            cv2.namedWindow(os.path.basename(image_path), 0)
            cv2.imshow(os.path.basename(image_path), image_with_bounding_boxes)
            cv2.waitKey(0)

        if args.method in GRAD_BASED_METHOD_MAP:
            model_wrapper.need_loss(False)
            cam_detector_visualizer.switch_activations_and_grads(model_wrapper)


if __name__ == '__main__':
    main()
