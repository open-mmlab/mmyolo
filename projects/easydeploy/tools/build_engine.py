import argparse

from ..model import EngineBuilder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--img-size',
        nargs='+',
        type=int,
        default=[640, 640],
        help='Image size of height and width')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='TensorRT builder device')
    parser.add_argument(
        '--scales',
        type=str,
        default='[[1,3,640,640],[1,3,640,640],[1,3,640,640]]',
        help='Input scales for build dynamic input shape engine')
    parser.add_argument(
        '--fp16', action='store_true', help='Build model with fp16 mode')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1
    return args


def main(args):
    img_size = (1, 3, *args.img_size)
    try:
        scales = eval(args.scales)
    except Exception:
        print('Input scales is not a python variable')
        print('Set scales default None')
        scales = None
    builder = EngineBuilder(args.checkpoint, img_size, args.device)
    builder.build(scales, fp16=args.fp16)


if __name__ == '__main__':
    args = parse_args()
    main(args)
