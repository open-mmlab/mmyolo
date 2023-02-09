# Copyright (c) OpenMMLab. All rights reserved.
import os
import urllib

import numpy as np
import torch
from mmengine.utils import scandir
from prettytable import PrettyTable

from mmyolo.models import RepVGGBlock

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def switch_to_deploy(model):
    """Model switch to deploy status."""
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    print('Switch model to deploy modality.')


def auto_arrange_images(image_list: list, image_column: int = 2) -> np.ndarray:
    """Auto arrange image to image_column x N row.

    Args:
        image_list (list): cv2 image list.
        image_column (int): Arrange to N column. Default: 2.
    Return:
        (np.ndarray): image_column x N row merge image
    """
    img_count = len(image_list)
    if img_count <= image_column:
        # no need to arrange
        image_show = np.concatenate(image_list, axis=1)
    else:
        # arrange image according to image_column
        image_row = round(img_count / image_column)
        fill_img_list = [np.ones(image_list[0].shape, dtype=np.uint8) * 255
                         ] * (
                             image_row * image_column - img_count)
        image_list.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(image_row):
            start_col = image_column * i
            end_col = image_column * (i + 1)
            merge_col = np.hstack(image_list[start_col:end_col])
            merge_imgs_col.append(merge_col)

        # merge to one image
        image_show = np.vstack(merge_imgs_col)

    return image_show


def get_file_list(source_root: str) -> [list, dict]:
    """Get file list.

    Args:
        source_root (str): image or video source path

    Return:
        source_file_path_list (list): A list for all source file.
        source_type (dict): Source type: file or url or dir.
    """
    is_dir = os.path.isdir(source_root)
    is_url = source_root.startswith(('http:/', 'https:/'))
    is_file = os.path.splitext(source_root)[-1].lower() in IMG_EXTENSIONS

    source_file_path_list = []
    if is_dir:
        # when input source is dir
        for file in scandir(source_root, IMG_EXTENSIONS, recursive=True):
            source_file_path_list.append(os.path.join(source_root, file))
    elif is_url:
        # when input source is url
        filename = os.path.basename(
            urllib.parse.unquote(source_root).split('?')[0])
        file_save_path = os.path.join(os.getcwd(), filename)
        print(f'Downloading source file to {file_save_path}')
        torch.hub.download_url_to_file(source_root, file_save_path)
        source_file_path_list = [file_save_path]
    elif is_file:
        # when input source is single image
        source_file_path_list = [source_root]
    else:
        print('Cannot find image file.')

    source_type = dict(is_dir=is_dir, is_url=is_url, is_file=is_file)

    return source_file_path_list, source_type


def show_data_classes(data_classes):
    """When printing an error, all class names of the dataset."""
    print('\n\nThe name of the class contained in the dataset:')
    data_classes_info = PrettyTable()
    data_classes_info.title = 'Information of dataset class'
    # List Print Settings
    # If the quantity is too large, 25 rows will be displayed in each column
    if len(data_classes) < 25:
        data_classes_info.add_column('Class name', data_classes)
    elif len(data_classes) % 25 != 0 and len(data_classes) > 25:
        col_num = int(len(data_classes) / 25) + 1
        data_name_list = list(data_classes)
        for i in range(0, (col_num * 25) - len(data_classes)):
            data_name_list.append('')
        for i in range(0, len(data_name_list), 25):
            data_classes_info.add_column('Class name',
                                         data_name_list[i:i + 25])

    # Align display data to the left
    data_classes_info.align['Class name'] = 'l'
    print(data_classes_info)


def is_metainfo_lower(cfg):
    """Determine whether the custom metainfo fields are all lowercase."""

    def judge_keys(dataloader_cfg):
        while 'dataset' in dataloader_cfg:
            dataloader_cfg = dataloader_cfg['dataset']
        if 'metainfo' in dataloader_cfg:
            all_keys = dataloader_cfg['metainfo'].keys()
            all_is_lower = all([str(k).islower() for k in all_keys])
            assert all_is_lower, f'The keys in dataset metainfo must be all lowercase, but got {all_keys}. ' \
                                 f'Please refer to https://github.com/open-mmlab/mmyolo/blob/e62c8c4593/configs/yolov5/yolov5_s-v61_syncbn_fast_1xb4-300e_balloon.py#L8' # noqa

    judge_keys(cfg.get('train_dataloader', {}))
    judge_keys(cfg.get('val_dataloader', {}))
    judge_keys(cfg.get('test_dataloader', {}))
