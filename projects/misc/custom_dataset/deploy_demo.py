import argparse

import torch
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='For mmdeploy predict')
    parser.add_argument('--deploy-cfg', help='deploy config path')
    parser.add_argument('--model-cfg', help='model config root')
    parser.add_argument(
        '--device', help='device used for conversion', default='cuda:0')
    parser.add_argument('--backend-model', help='backend model path')
    parser.add_argument('--img', help='image path')
    args = parser.parse_args()
    return args


def deploy_demo(args):
    backend_model = [args.backend_model]
    image = args.img

    # read deploy_cfg and model_cfg
    deploy_cfg, model_cfg = load_config(args.deploy_cfg, args.model_cfg)

    # build task and backend model
    task_processor = build_task_processor(model_cfg, deploy_cfg, args.device)
    model = task_processor.build_backend_model(backend_model)

    # process input image
    input_shape = get_input_shape(deploy_cfg)
    model_inputs, _ = task_processor.create_input(image, input_shape)

    # do model inference
    with torch.no_grad():
        result = model.test_step(model_inputs)

    # visualize results
    task_processor.visualize(
        image=image,
        model=model,
        result=result[0],
        window_name='visualize',
        output_file='output_detection.png')


def main():
    args = parse_args()
    deploy_demo(args)
    print('All done!')


if __name__ == '__main__':
    main()
