# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmengine.config import Config, DictAction
from mmengine.dataset import COLLATE_FUNCTIONS
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner

from mmyolo.datasets import YOLOv5CocoDataset
from mmyolo.registry import RUNNERS, MODELS, DATASETS, VISUALIZERS
from mmyolo.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMYOLO test (and eval) a model')
    parser.add_argument('config', help='test config file path')
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


def main():
    args = parse_args()
    register_all_modules()

    assert 'yolov5' in args.config, 'Now, this script only support yolov5.'

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = MODELS.build(cfg.model)
    model.eval()
    dataset_cfg = cfg.get('train_dataloader').get('dataset')
    dataset = DATASETS.build(dataset_cfg)    # type: YOLOv5CocoDataset

    # get collate_fn
    collate_fn_cfg = cfg.get('train_dataloader').pop('collate_fn',
                                                     dict(type='pseudo_collate'))
    collate_fn_type = collate_fn_cfg.pop('type')
    collate_fn = COLLATE_FUNCTIONS.get(collate_fn_type)

    # init visualizer
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    idx = 0

    data_info = dataset.get_data_info(idx)
    data = dataset.prepare_data(idx)

    batch_data = collate_fn([data])
    model.assign(batch_data)





    # mmcv.imshow(data['inputs'])
    #
    # pass



if __name__ == '__main__':
    main()