# Copyright (c) OpenMMLab. All rights reserved.
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
        'https://mmdeploy.readthedocs.io/en/latest/01-how-to-build/build_from_source.html'
    )  # noqa


def parse_args():
    parser = argparse.ArgumentParser(description='For mmdeploy predict')
    parser.add_argument('img', help='image path')
    parser.add_argument('--model-cfg', help='model config root')
    parser.add_argument('--backend-model', help='backend model path')
    parser.add_argument('--deploy-cfg', help='deploy config path')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', help='device used for conversion', default='cuda:0')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # read deploy_cfg and model_cfg
    deploy_cfg, model_cfg = load_config(args.deploy_cfg, args.model_cfg)

    # build task and backend model
    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)
    model = task_processor.build_backend_model([args.backend_model])

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
        out_file = os.path.join(args.out_dir, filename)

        # visualize results
        task_processor.visualize(
            image=file,
            model=model,
            result=result[0],
            window_name='visualize',
            output_file=out_file)

        progress_bar.update()

    print('All done!')


if __name__ == '__main__':
    main()
