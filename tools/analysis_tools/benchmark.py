# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import time

import torch
from mmengine import Config, DictAction
from mmengine.dist import get_world_size, init_dist
from mmengine.logging import MMLogger, print_log
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist
from mmengine.utils.dl_utils import set_multi_processing

from mmyolo.registry import MODELS


# TODO: Refactoring and improving
def parse_args():
    parser = argparse.ArgumentParser(description='MMYOLO benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--repeat-num',
        type=int,
        default=1,
        help='number of repeat times of measurement for averaging the results')
    parser.add_argument(
        '--max-iter', type=int, default=2000, help='num of max iter')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing '
        'benchmark metrics')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def measure_inference_speed(cfg, checkpoint, max_iter, log_interval,
                            is_fuse_conv_bn):
    env_cfg = cfg.get('env_cfg')
    if env_cfg.get('cudnn_benchmark'):
        torch.backends.cudnn.benchmark = True

    mp_cfg: dict = env_cfg.get('mp_cfg', {})
    set_multi_processing(**mp_cfg, distributed=cfg.distributed)

    # Because multiple processes will occupy additional CPU resources,
    # FPS statistics will be more unstable when num_workers is not 0.
    # It is reasonable to set num_workers to 0.
    dataloader_cfg = cfg.test_dataloader
    dataloader_cfg['num_workers'] = 0
    dataloader_cfg['batch_size'] = 1
    dataloader_cfg['persistent_workers'] = False
    data_loader = Runner.build_dataloader(dataloader_cfg)

    # build the model and load checkpoint
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = model.cuda()
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model.test_step(data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print_log(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img', 'current')

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print_log(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img', 'current')
            break
    return fps


def repeat_measure_inference_speed(cfg,
                                   checkpoint,
                                   max_iter,
                                   log_interval,
                                   is_fuse_conv_bn,
                                   repeat_num=1):
    assert repeat_num >= 1

    fps_list = []

    for _ in range(repeat_num):
        cp_cfg = copy.deepcopy(cfg)

        fps_list.append(
            measure_inference_speed(cp_cfg, checkpoint, max_iter, log_interval,
                                    is_fuse_conv_bn))

    if repeat_num > 1:
        fps_list_ = [round(fps, 1) for fps in fps_list]
        times_pre_image_list_ = [round(1000 / fps, 1) for fps in fps_list]
        mean_fps_ = sum(fps_list_) / len(fps_list_)
        mean_times_pre_image_ = sum(times_pre_image_list_) / len(
            times_pre_image_list_)
        print_log(
            f'Overall fps: {fps_list_}[{mean_fps_:.1f}] img / s, '
            f'times per image: '
            f'{times_pre_image_list_}[{mean_times_pre_image_:.1f}] ms / img',
            'current')
        return fps_list

    return fps_list[0]


# TODO: refactoring
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmyolo'))

    distributed = False
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.get('env_cfg', {}).get('dist_cfg', {}))
        distributed = True
        assert get_world_size(
        ) == 1, 'Inference benchmark does not allow distributed multi-GPU'

    cfg.distributed = distributed

    log_file = None
    if args.work_dir:
        log_file = os.path.join(args.work_dir, 'benchmark.log')
        mkdir_or_exist(args.work_dir)

    MMLogger.get_instance('mmyolo', log_file=log_file, log_level='INFO')

    repeat_measure_inference_speed(cfg, args.checkpoint, args.max_iter,
                                   args.log_interval, args.fuse_conv_bn,
                                   args.repeat_num)


if __name__ == '__main__':
    main()
