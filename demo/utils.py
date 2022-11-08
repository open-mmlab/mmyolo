# Copyright (c) OpenMMLab. All rights reserved.

import os
import urllib

import mmcv
import numpy as np
import torch
from mmengine.utils import scandir

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def auto_arrange_images(image_list: list, image_column: int = 2) -> np.ndarray:
    """Auto arrange image to 2 x N.

    Args:
        image_list (list): cv2 image list.
        image_column (int): Arrange to N column.
    Return:
        (np.ndarray): image_column x N merge image
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


def get_file_list(source_root: str) -> [list, bool, bool, bool]:
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
        torch.hub.download_url_to_file(source_root, filename)
        source_file_path_list = [os.path.join(os.getcwd(), filename)]
    elif is_file:
        # when input source is single image
        source_file_path_list = [source_root]
    else:
        print('Cannot find image file.')

    source_type = dict(is_dir=is_dir, is_url=is_url, is_file=is_file)

    return source_file_path_list, source_type


def get_image_and_out_file_path(source_path: str,
                                source_root: str,
                                is_dir: bool,
                                out_dir: str,
                                show_flag: bool = False) -> [np.ndarray, str]:
    """Get file list.

    Args:
        source_path (str): Path of source file
        source_root (str): Path of source root
        is_dir (bool): Source root is directory or not
        out_dir (str): Output save path
        show_flag (bool): Whether show the result or not

    Return:
        img (np.ndarray): cv2 file data
        out_file_path (str): Out file save path according of it original path
    """

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    img = mmcv.imread(source_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    if is_dir:
        filename = os.path.relpath(source_path, source_root).replace('/', '_')
    else:
        filename = os.path.basename(source_path)
    out_file_path = None if show_flag else os.path.join(out_dir, filename)

    return img, out_file_path
