"""This script helps to convert yolo-style dataset to the coco format.

Usage:
    $ python yolo2coco.py --image-dir /path/to/dataset   # the root dir
                          [--split]                      # if splits

Note:
    1. Before running this script, please make sure the root directory
    of your dataset is formatted in the following struction:
            |____ $ROOT_PATH
            |__ class.txt
            |__ labels
            |__ images
            |__ ...
    2. `-split` is not used by default. If you need to use it, please ensure
    the corresponding`train.txt`, ` val.txt`, and `test.txt` must exist under
    `-image-dir`. Otherwise, the script will fail to run.
    3. Once the script finishes, the result files will be saved in the
    directory named 'coco_format' in the root directory of your dataset.
"""
import argparse
import os
import os.path as osp

import mmcv
import mmengine


def check_existance(file_path: str):
    """Check if target file is existed."""
    if not osp.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist!')


def get_image_info(yolo_image_dir, idx, file_name):
    """Retrieve image information."""
    img_path = osp.join(yolo_image_dir, file_name)
    check_existance(img_path)

    img = mmcv.imread(img_path)
    height, width = img.shape[:2]
    img_info_dict = {
        'file_name': file_name,
        'id': idx,
        'width': width,
        'height': height
    }

    return img_info_dict, height, width


def convert_bbox_info(label, idx, obj_count, H, W):
    """Convert yolo-style bbox info to the coco format."""
    label = label.strip().split()
    x = float(label[1])
    y = float(label[2])
    w = float(label[3])
    h = float(label[4])

    # convert x,y,w,h to x1,y1,x2,y2
    x1 = (x - w / 2) * W
    y1 = (y - h / 2) * H
    x2 = (x + w / 2) * W
    y2 = (y + h / 2) * H

    cls_id = int(label[0])
    width = max(0., x2 - x1)
    height = max(0., y2 - y1)
    coco_format_info = {
        'image_id': idx,
        'id': obj_count,
        'category_id': cls_id,
        'bbox': [x1, y1, width, height],
        'area': width * height,
        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
        'iscrowd': 0
    }

    return coco_format_info, obj_count


def organize_by_existing_files(image_dir: str):
    """Format annotations by existing train/val/test files."""
    categories = ['train', 'val', 'test']
    image_list = []

    for category in categories:
        txt_file = osp.join(image_dir, f'{category}.txt')
        print(f'Start to read {category} dataset definition from {txt_file}')
        assert osp.exists(txt_file)

        with open(txt_file) as f:
            image_path = f.readlines()
            image_list.append(image_path)
    return image_list[0], image_list[1], image_list[2]


def convert_yolo_to_coco(image_dir: str, split: bool = False):
    """Convert annotations from yolo style to coco style.

    Args:
        image_dir (str): the root directory of your datasets which contains
            labels, images, classes.txt, etc
        split (bool): whether to organize the datasets based on existing
            train.txt, val.txt, and test.txt
    """
    print(f'Start to load existing images and annotations from {image_dir}')
    check_existance(image_dir)

    yolo_label_dir = osp.join(image_dir, 'labels')
    yolo_image_dir = osp.join(image_dir, 'images')
    yolo_class_txt = osp.join(image_dir, 'classes.txt')
    check_existance(yolo_label_dir)
    check_existance(yolo_image_dir)
    check_existance(yolo_class_txt)

    with open(yolo_class_txt) as f:
        classes = f.read().strip().split()

    indices = os.listdir(yolo_image_dir)

    if split:
        print('Start to work based to existing train, test, and val')
        train_dataset = {'images': [], 'annotations': [], 'categories': []}
        val_dataset = {'images': [], 'annotations': [], 'categories': []}
        test_dataset = {'images': [], 'annotations': [], 'categories': []}

        # categories id starts from 0
        for i, cls in enumerate(classes, 0):
            train_dataset['categories'].append({'id': i, 'name': cls})
            val_dataset['categories'].append({'id': i, 'name': cls})
            test_dataset['categories'].append({'id': i, 'name': cls})
        train_img, val_img, test_img = organize_by_existing_files(image_dir)
    else:
        print('Start to work on all data')
        dataset = {'images': [], 'annotations': [], 'categories': []}
        for i, cls in enumerate(classes, 0):
            dataset['categories'].append({'id': i, 'name': cls})

    obj_count = 0
    for idx, image in enumerate(mmengine.track_iter_progress(indices)):
        """Comments:
        1、replace may cause bug, while img filename is images1.jpg.
            Try to use os.path.splitext.
        2、It is better to use a format list like ['.jpg', '.png', '.jpeg']
            to verify file postfixes.
        3、use .lower() while verify file postfixes.
        """
        # support both .jpg, .png, and .jpeg
        img_suffix = osp.splitext(image)[1].lower()
        if img_suffix in ['.jpg', '.png', '.jpeg']:
            img_name = osp.splitext(image)[0]
        img_info_dict, H, W = get_image_info(yolo_image_dir, idx, image)

        if split:
            if image in train_img:
                dataset = train_dataset
            elif image in val_img:
                dataset = val_dataset
            elif image in test_img:
                dataset = test_dataset

        dataset['images'].append(img_info_dict)

        label_path = osp.join(yolo_label_dir, img_name) + '.txt'
        if not osp.exists(label_path):
            # if current image is not annotated
            print(f'WARNING: {label_path} does not exist. Skipped.')
            continue

        with open(label_path) as f:
            labels = f.readlines()
            for label in labels:
                coco_info, obj_count = convert_bbox_info(
                    label, idx, obj_count, H, W)
                dataset['annotations'].append(coco_info)
                obj_count += 1

    # saving results to result json in res_folder
    res_folder = osp.join(image_dir, 'coco_format')
    if not osp.exists(res_folder):
        os.makedirs(res_folder)

    if split:
        for category in ['train', 'val', 'test']:
            out_file = osp.join(image_dir, f'coco_format/{category}.json')
            print(f'Saving converted annotations to {out_file}')
            if category == 'train':
                mmengine.dump(train_dataset, out_file)
            elif category == 'val':
                mmengine.dump(val_dataset, out_file)
            elif category == 'test':
                mmengine.dump(test_dataset, out_file)
    else:
        out_file = osp.join(image_dir, 'coco_format/result.json')
        print(f'Saving converted annotations to {out_file}')
        mmengine.dump(dataset, out_file)
    print(f'Conversion is finished! Please check at {res_folder}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-dir',
        type=str,
        required=True,
        help='dataset directory with ./images and ./labels, classes.txt, etc.')
    parser.add_argument(
        '--split',
        action='store_true',
        help='convert based on existing train.txt, val.txt, and test.txt')
    arg = parser.parse_args()
    convert_yolo_to_coco(arg.image_dir, arg.split)
