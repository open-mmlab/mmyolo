import argparse
import os.path as osp
import shutil
from collections import defaultdict

import mmengine
import numpy as np
from pycocotools.coco import COCO

try:
    import panopticapi
except ImportError:
    panopticapi = None


class COCOPanoptic(COCO):
    """This wrapper is for loading the panoptic style annotation file.

    The format is shown in the CocoPanopticDataset class.

    Args:
        annotation_file (str, optional): Path of annotation file.
            Defaults to None.
    """

    def __init__(self, annotation_file=None) -> None:
        super().__init__(annotation_file)

    def createIndex(self) -> None:
        """Create index."""
        # create index
        print('creating index...')
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    # to match with instance.json
                    seg_ann['image_id'] = ann['image_id']
                    img_to_anns[ann['image_id']].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    # annotations from different images but
                    # may have same segment_id
                    if seg_ann['id'] in anns.keys():
                        anns[seg_ann['id']].append(seg_ann)
                    else:
                        anns[seg_ann['id']] = [seg_ann]

            # filter out annotations from other images
            img_to_anns_ = defaultdict(list)
            for k, v in img_to_anns.items():
                img_to_anns_[k] = [x for x in v if x['image_id'] == k]
            img_to_anns = img_to_anns_

        if 'images' in self.dataset:
            for img_info in self.dataset['images']:
                img_info['segm_file'] = img_info['file_name'].replace(
                    'jpg', 'png')
                imgs[img_info['id']] = img_info

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats


def _process_data(args, in_dataset_type: str, out_dataset_type: str):
    assert in_dataset_type in ('train', 'val')
    assert out_dataset_type in ('train', 'val')

    year = '2017'
    with_panoptic = args.panoptic

    int_ann_file_name = f'annotations/instances_{in_dataset_type}{year}.json'
    out_ann_file_name = f'annotations/instances_{out_dataset_type}{year}.json'

    ann_path = osp.join(args.root, int_ann_file_name)
    json_data = mmengine.load(ann_path)

    new_json_data = {
        'info': json_data['info'],
        'licenses': json_data['licenses'],
        'categories': json_data['categories'],
        'images': [],
        'annotations': []
    }

    images = json_data['images']
    coco = COCO(ann_path)

    if with_panoptic:
        in_panoptic_ann_file_name = \
            f'annotations/panoptic_{in_dataset_type}{year}.json'
        out_panoptic_ann_file_name = \
            f'annotations/panoptic_{out_dataset_type}{year}.json'

        ann_path = osp.join(args.root, in_panoptic_ann_file_name)
        json_data = mmengine.load(ann_path)
        panoptic_new_json_data = {
            'info': json_data['info'],
            'licenses': json_data['licenses'],
            'categories': json_data['categories'],
            'images': [],
            'annotations': []
        }

        panoptic_ann_path = osp.join(args.root, in_panoptic_ann_file_name)
        coco_panoptic = COCOPanoptic(panoptic_ann_path)

    # shuffle
    np.random.shuffle(images)

    progress_bar = mmengine.ProgressBar(args.num_img)

    for i in range(args.num_img):
        file_name = images[i]['file_name']
        stuff_file_name = osp.splitext(file_name)[0] + '.png'
        image_path = osp.join(args.root, in_dataset_type + year, file_name)
        stuff_image_path = osp.join(args.root, 'stuffthingmaps',
                                    in_dataset_type + year, stuff_file_name)

        ann_ids = coco.getAnnIds(imgIds=[images[i]['id']])
        ann_info = coco.loadAnns(ann_ids)

        new_json_data['images'].append(images[i])
        new_json_data['annotations'].extend(ann_info)

        if with_panoptic:
            panoptic_stuff_image_path = osp.join(
                args.root, 'annotations', 'panoptic_' + in_dataset_type + year,
                stuff_file_name)
            panoptic_ann_info = coco_panoptic.imgToAnns.get(images[i]['id'])
            ann_dict = {
                'segments_info': panoptic_ann_info,
                'file_name': stuff_file_name,
                'image_id': images[i]['id']
            }
            panoptic_new_json_data['images'].append(images[i])
            panoptic_new_json_data['annotations'].append(ann_dict)

            if osp.exists(panoptic_stuff_image_path):
                shutil.copy(
                    panoptic_stuff_image_path,
                    osp.join(args.out_dir, 'annotations',
                             'panoptic_' + out_dataset_type + year,
                             stuff_file_name))

        shutil.copy(image_path, osp.join(args.out_dir,
                                         out_dataset_type + year))
        if osp.exists(stuff_image_path):
            shutil.copy(
                stuff_image_path,
                osp.join(args.out_dir, 'stuffthingmaps',
                         out_dataset_type + year))

        progress_bar.update()

    mmengine.dump(new_json_data, osp.join(args.out_dir, out_ann_file_name))

    if with_panoptic:
        mmengine.dump(panoptic_new_json_data,
                      osp.join(args.out_dir, out_panoptic_ann_file_name))


def _make_dirs(out_dir, with_panoptic: bool = False):
    mmengine.mkdir_or_exist(out_dir)
    mmengine.mkdir_or_exist(osp.join(out_dir, 'annotations'))
    mmengine.mkdir_or_exist(osp.join(out_dir, 'train2017'))
    mmengine.mkdir_or_exist(osp.join(out_dir, 'val2017'))
    mmengine.mkdir_or_exist(osp.join(out_dir, 'stuffthingmaps/train2017'))
    mmengine.mkdir_or_exist(osp.join(out_dir, 'stuffthingmaps/val2017'))
    mmengine.mkdir_or_exist(osp.join(out_dir, 'stuffthingmaps/val2017'))

    if with_panoptic:
        mmengine.mkdir_or_exist(
            osp.join(out_dir, 'annotations/panoptic_train2017'))
        mmengine.mkdir_or_exist(
            osp.join(out_dir, 'annotations/panoptic_val2017'))


def parse_args():
    parser = argparse.ArgumentParser(description='Extract coco subset')
    parser.add_argument('root', help='root path')
    parser.add_argument(
        'out_dir', type=str, help='directory where subset coco will be saved.')
    parser.add_argument(
        '--num-img', default=50, type=int, help='num of extract image')
    parser.add_argument(
        '--use-training-set',
        action='store_true',
        help='Whether to use the training set when extract the training set. '
        'The training subset is extracted from the validation set by '
        'default which can speed up.')
    parser.add_argument(
        '--panoptic',
        action='store_true',
        help='Support to process panoptic dataset')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out_dir != args.root, \
        'The file will be overwritten in place, ' \
        'so the same folder is not allowed !'

    _make_dirs(args.out_dir, args.panoptic)

    print('start processing train dataset')
    if args.use_training_set:
        _process_data(args, 'train', 'train')
    else:
        _process_data(args, 'val', 'train')
    print('start processing val dataset')
    _process_data(args, 'val', 'val')


if __name__ == '__main__':
    main()
