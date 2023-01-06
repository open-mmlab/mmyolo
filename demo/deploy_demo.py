# Copyright (c) OpenMMLab. All rights reserved.
"""Deploy demo for mmdeploy.

This script help user to run mmdeploy demo after convert the
checkpoint to backends.

Usage:
    python deploy_demo.py img \
                          config \
                          checkpoint \
                          [--deploy-cfg DEPLOY_CFG] \
                          [--device DEVICE] \
                          [--out-dir OUT_DIR] \
                          [--show] \
                          [--score-thr SCORE_THR]

Example:
    python deploy_demo.py \
        ${MMYOLO_PATH}/data/cat/images \
        ./yolov5_s-v61_syncbn_fast_1xb32-100e_cat.py \
        ./end2end.engine \
        --deploy-cfg ./detection_tensorrt-fp16_dynamic-192x192-960x960.py \
        --out-dir ${MMYOLO_PATH}/work_dirs/deploy_predict_out \
        --device cuda:0 \
        --score-thr 0.5
"""
import argparse
import os

import torch
from mmengine import ProgressBar

from mmyolo.utils.misc import get_file_list

try:
    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import get_input_shape, load_config
except ImportError:
    raise ImportError(
        'mmdeploy is not installed, please see '
        'https://mmdeploy.readthedocs.io/en/1.x/01-how-to-build/build_from_source.html'  # noqa
    )


def parse_args():
    parser = argparse.ArgumentParser(description='For mmdeploy predict')
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='model config root')
    parser.add_argument('checkpoint', help='checkpoint backend model path')
    parser.add_argument('--deploy-cfg', help='deploy config path')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for conversion')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    args = parser.parse_args()
    return args


# TODO Still need to refactor to not building dataset.
def main():
    args = parse_args()

    if not os.path.exists(args.out_dir) and not args.show:
        os.mkdir(args.out_dir)

    # read deploy_cfg and config
    deploy_cfg, model_cfg = load_config(args.deploy_cfg, args.config)

    # build task and backend model
    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)
    model = task_processor.build_backend_model([args.checkpoint])

    # get model input shape
    input_shape = get_input_shape(deploy_cfg)

    # get file list
    files, source_type = get_file_list(args.img)

    # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        # process input image
        model_inputs, _ = task_processor.create_input(file, input_shape)

        # do model inference
        with torch.no_grad():
            result = model.test_step(model_inputs)

        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        # filter score
        result = result[0]
        result.pred_instances = result.pred_instances[
            result.pred_instances.scores > args.score_thr]

        # visualize results
        task_processor.visualize(
            image=file,
            model=model,
            result=result,
            show_result=args.show,
            window_name=os.path.basename(filename),
            output_file=out_file)

        progress_bar.update()

    print('All done!')


if __name__ == '__main__':
    main()
