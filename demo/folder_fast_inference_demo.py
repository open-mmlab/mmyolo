# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from mmcv.transforms import Compose
from mmdet.apis import DetInferencer
from mmyolo.utils import switch_to_deploy
from mmyolo.utils.misc import get_file_list
from rich.progress import track

try:
    from mmdet.datasets.api_wrappers import COCO
except ImportError:
    COCO = None


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'root', help='data root dir or coco json file path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--batch-size', type=int, default=8, help='Inference batch size.')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--no-save-img', action='store_true', help='Whether not to save the prediction img results')
    parser.add_argument(
        '--save-coco', action='store_true', help='Whether to save the prediction results in json format')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    args = parser.parse_args()
    return args


class FastDetInferencer(DetInferencer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, scope='mmyolo', **kwargs)
        switch_to_deploy(self.model)

    def _init_pipeline(self, cfg) -> Compose:
        return Compose(cfg.inference_only_pipeline)

    def _inputs_to_list(self, path) -> list:
        # get file list
        if path.endswith('.json'):
            if COCO is None:
                raise RuntimeError('Please install cocoapi to use coco json file by "pip install pycocotools".')
            coco = COCO(path)
            img_ids = coco.get_img_ids()
            total_img_infos = []
            for img_id in img_ids:
                raw_img_info = coco.load_imgs([img_id])[0]
                raw_img_info['img_id'] = img_id
                total_img_infos.append(raw_img_info)
        else:
            total_img_infos, _ = get_file_list(path)

        return total_img_infos

    def __call__(self,
                 inputs,
                 batch_size: int = 1,
                 pred_score_thr: float = 0.3,
                 save_pred_img: bool = True,
                 save_coco: bool = False,
                 out_dir: str = '',
                 **kwargs) -> None:
        ori_inputs = self._inputs_to_list(inputs)
        inputs = self.preprocess(ori_inputs, batch_size=batch_size)
        for ori_inputs, data in track(inputs, description='Inference'):
            preds = self.forward(data)
            print(preds)
            if save_pred_img:
                self.save_pred_img(ori_inputs, preds, out_dir)
            if save_coco:
                self.save_coco(ori_inputs, preds, out_dir)

    def save_pred_img(self, ori_inputs, preds, out_dir):
        pass

    def save_coco(self, ori_inputs, preds, out_dir):
        pass


def main():
    args = parse_args()

    if args.save_coco:
        assert args.root.endswith('.json'), \
            'If you want to save coco format result, please use coco json file as input.'

    inferencer = FastDetInferencer(args.config, args.checkpoint, device=args.device)
    inferencer(args.root, batch_size=args.batch_size,
               pred_score_thr=args.score_thr,
               save_pred_img=not args.no_save_img,
               save_coco=not args.save_coco,
               out_dir=args.out_dir)


if __name__ == '__main__':
    main()
