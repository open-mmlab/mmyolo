import argparse
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download datasets for training')
    parser.add_argument(
        '--dataset-name', type=str, help='dataset name', default='coco2017')
    parser.add_argument(
        '--save-dir',
        type=str,
        help='the dir to save dataset',
        default='data/coco')
    parser.add_argument(
        '--unzip',
        action='store_true',
        help='whether unzip dataset or not, zipped files will be saved')
    parser.add_argument(
        '--delete',
        action='store_true',
        help='delete the download zipped files')
    parser.add_argument(
        '--threads', type=int, help='number of threading', default=4)
    args = parser.parse_args()
    return args


def download(url, dir, unzip=True, delete=False, threads=1):

    def download_one(url, dir):
        f = dir / Path(url).name
        if Path(url).is_file():
            Path(url).rename(f)
        elif not f.exists():
            print(f'Downloading {url} to {f}')
            torch.hub.download_url_to_file(url, f, progress=True)
        if unzip and f.suffix in ('.zip', '.tar'):
            print(f'Unzipping {f.name}')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)
            elif f.suffix == '.tar':
                TarFile(f).extractall(path=dir)
            if delete:
                f.unlink()
                print(f'Delete {f}')

    dir = Path(dir)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def main():
    args = parse_args()
    path = Path(args.save_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    data2url = dict(
        # TODO: Support for downloading Panoptic Segmentation of COCO
        coco2017=[
            'http://images.cocodataset.org/zips/train2017.zip',
            'http://images.cocodataset.org/zips/val2017.zip',
            'http://images.cocodataset.org/zips/test2017.zip',
            'http://images.cocodataset.org/annotations/' +
            'annotations_trainval2017.zip'
        ],
        lvis=[
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip',  # noqa
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip',  # noqa
        ],
        voc2007=[
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',  # noqa
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',  # noqa
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar',  # noqa
        ],
        voc2012=[
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',  # noqa
        ],
        balloon=[
            # src link: https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip # noqa
            'https://download.openmmlab.com/mmyolo/data/balloon_dataset.zip'
        ],
        cat=[
            'https://download.openmmlab.com/mmyolo/data/cat_dataset.zip'  # noqa
        ],
        visdrone2019=[
            'https://bj.bcebos.com/v1/ai-studio-online/e6443b5fcb79447b80bbdb0c8dc10ed0763921bfbca84a93bc11acdc3bea6acb?responseContentDisposition=attachment%3B%20filename%3Dvisdrone.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2022-09-18T07%3A25%3A55Z%2F-1%2F%2F104606cd90872c0bd13c4ad23646b1fed78cbf9ab11925b2b3996648115ad5b6'  # noqa
        ],
    )
    url = data2url.get(args.dataset_name, None)
    if url is None:
        print('Only support COCO, VOC, balloon, cat and LVIS now!')
        return
    download(
        url,
        dir=path,
        unzip=args.unzip,
        delete=args.delete,
        threads=args.threads)


if __name__ == '__main__':
    main()
