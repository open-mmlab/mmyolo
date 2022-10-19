"""This script helps to convert yolo-style dataset to the coco format.

Usage:
    $ python yolo2coco.py --image-dir /path/to/dataset   # the root dir

Note:
    1. Before running this script, please make sure the root directory
    of your dataset is formatted in the following struction:
    .
    └── $ROOT_PATH
        ├── classes.txt
        ├── labels
        │    ├── a.txt
        │    ├── b.txt
        │    └── ...
        ├── images
        │    ├── a.jpg
        │    ├── b.png
        │    └── ...
        └── ...
    2. The script will automatically check whether the corresponding
    `train.txt`, ` val.txt`, and `test.txt` exist under `image-dir` or not.
    If these files are detected, the script will organize the dataset.
    The image paths in these files must be ABSOLUTE paths.
    3. Once the script finishes, the result files will be saved in the
    directory named 'annotations' in the root directory of your dataset. The
    corresponding images will be kept in `.jpg` format. The 'annotations'
    folder may look like this in the root directory:
    .
    └── $ROOT_PATH
        ├── annotations
        │    ├── train.json
        │    ├── train
        │    |    └── img1.jpg
             |    └── ...
        │    └── ...
        └── ...
"""
import argparse
import os
import os.path as osp
import shutil

import mmcv
import mmengine


def check_existence(file_path: str):
    """Check if target file is existed."""
    if not osp.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist!')


def get_image_info(yolo_image_dir, idx, file_name):
    """Retrieve image information."""
    img_path = osp.join(yolo_image_dir, file_name)
    check_existence(img_path)

    img = mmcv.imread(img_path)
    height, width = img.shape[:2]
    img_info_dict = {
        'file_name': file_name,
        'id': idx,
        'width': width,
        'height': height
    }

    return img_info_dict, height, width


def convert_bbox_info(label, idx, obj_count, image_height, image_width):
    """Convert yolo-style bbox info to the coco format."""
    label = label.strip().split()
    x = float(label[1])
    y = float(label[2])
    w = float(label[3])
    h = float(label[4])

    # convert x,y,w,h to x1,y1,x2,y2
    x1 = (x - w / 2) * image_width
    y1 = (y - h / 2) * image_height
    x2 = (x + w / 2) * image_width
    y2 = (y + h / 2) * image_height

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


def organize_by_existing_files(image_dir: str, existing_cat: list):
    """Format annotations by existing train/val/test files."""
    categories = ['train', 'val', 'test']
    image_list = []

    for cat in categories:
        if cat in existing_cat:
            txt_file = osp.join(image_dir, f'{cat}.txt')
            print(f'Start to read {cat} dataset definition')
            assert osp.exists(txt_file)

            with open(txt_file) as f:
                img_paths = f.readlines()
                img_paths = [
                    os.path.split(img_path.strip())[1]
                    for img_path in img_paths
                ]  # split the absolute path
                image_list.append(img_paths)
        else:
            image_list.append([])
    return image_list[0], image_list[1], image_list[2]


def copy_image_categorically(image_dir, output_folder, category, image,
                             img_suffix):
    """Allocate the images according to their categories in separate folder."""
    cat_folder = osp.join(output_folder, category)
    check_existence(cat_folder)

    img_path = osp.join(image_dir, 'images', image)
    img_name = osp.splitext(image)[0]
    img_data = mmcv.imread(str(img_path))
    new_img_name = f'{img_name}.jpg'
    new_img_path = osp.join(cat_folder, new_img_name)

    if img_suffix == '.jpg':
        shutil.copyfile(img_path, new_img_path)
    else:
        mmcv.imwrite(img_data, new_img_path)
    check_existence(new_img_path)


def convert_yolo_to_coco(image_dir: str):
    """Convert annotations from yolo style to coco style.

    Args:
        image_dir (str): the root directory of your datasets which contains
            labels, images, classes.txt, etc
    """
    print(f'Start to load existing images and annotations from {image_dir}')
    check_existence(image_dir)

    # check local environment
    yolo_label_dir = osp.join(image_dir, 'labels')
    yolo_image_dir = osp.join(image_dir, 'images')
    yolo_class_txt = osp.join(image_dir, 'classes.txt')
    check_existence(yolo_label_dir)
    check_existence(yolo_image_dir)
    check_existence(yolo_class_txt)
    print(f'All necessary files are located at {image_dir}')

    train_txt_path = osp.join(image_dir, 'train.txt')
    val_txt_path = osp.join(image_dir, 'val.txt')
    test_txt_path = osp.join(image_dir, 'test.txt')
    train_txt_found, val_txt_found, test_txt_found = False, False, False
    target_categories = []
    print(f'Checking if train.txt, val.txt, and test.txt are in {image_dir}')
    if osp.exists(train_txt_path):
        print('Found train.txt')
        train_txt_found = True
        target_categories.append('train')
    if osp.exists(val_txt_path):
        print('Found val.txt')
        val_txt_found = True
        target_categories.append('val')
    if osp.exists(test_txt_path):
        print('Found test.txt')
        test_txt_found = True
        target_categories.append('test')

    if train_txt_found or val_txt_found or test_txt_found:
        print('Need to organize the data accordingly.')
        to_categorize = True
    else:
        print('These files are not located, no need to organize separately.')
        to_categorize = False

    # prepare the output folders
    output_folder = osp.join(image_dir, 'annotations')
    if not osp.exists(output_folder):
        os.makedirs(output_folder)
    for category in target_categories:
        cat_out_folder = osp.join(output_folder, category)
        if not osp.exists(cat_out_folder):
            os.makedirs(cat_out_folder)

    # start the conversion and allocate images
    with open(yolo_class_txt) as f:
        classes = f.read().strip().split()

    indices = os.listdir(yolo_image_dir)

    dataset = {'images': [], 'annotations': [], 'categories': []}
    if to_categorize:
        train_dataset = {'images': [], 'annotations': [], 'categories': []}
        val_dataset = {'images': [], 'annotations': [], 'categories': []}
        test_dataset = {'images': [], 'annotations': [], 'categories': []}

        # categories id starts from 0
        for i, cls in enumerate(classes, 0):
            train_dataset['categories'].append({'id': i, 'name': cls})
            val_dataset['categories'].append({'id': i, 'name': cls})
            test_dataset['categories'].append({'id': i, 'name': cls})
        train_img, val_img, test_img = organize_by_existing_files(
            image_dir, target_categories)
    else:
        for i, cls in enumerate(classes, 0):
            dataset['categories'].append({'id': i, 'name': cls})

    obj_count = 0
    for idx, image in enumerate(mmengine.track_iter_progress(indices)):
        # support both .jpg, .png, and .jpeg
        img_suffix = osp.splitext(image)[1].lower()
        if img_suffix not in ['.jpg', '.png', '.jpeg']:
            raise Exception(
                "Only supports '.jpg', '.png', and '.jpeg' image formats")
        img_name = osp.splitext(image)[0]
        img_info_dict, image_height, image_width = get_image_info(
            yolo_image_dir, idx, image)

        if to_categorize:
            if image in train_img:
                dataset = train_dataset
                copy_image_categorically(image_dir, output_folder, 'train',
                                         image, img_suffix)
            elif image in val_img:
                dataset = val_dataset
                copy_image_categorically(image_dir, output_folder, 'val',
                                         image, img_suffix)
            elif image in test_img:
                dataset = test_dataset
                copy_image_categorically(image_dir, output_folder, 'test',
                                         image, img_suffix)

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
                    label, idx, obj_count, image_height, image_width)
                dataset['annotations'].append(coco_info)
                obj_count += 1

    # saving results to result json in res_folder
    if to_categorize:
        for category in target_categories:
            out_file = osp.join(output_folder, f'{category}.json')
            print(f'Saving converted results to {out_file}')
            if category == 'train':
                mmengine.dump(train_dataset, out_file)
            elif category == 'val':
                mmengine.dump(val_dataset, out_file)
            elif category == 'test':
                mmengine.dump(test_dataset, out_file)
    else:
        out_file = osp.join(image_dir, 'annotations/result.json')
        print(f'Saving converted results to {out_file}')
        mmengine.dump(dataset, out_file)
    print(f'All data has been converted. Please check at {output_folder} !')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-dir',
        type=str,
        required=True,
        help='dataset directory with ./images and ./labels, classes.txt, etc.')
    arg = parser.parse_args()
    convert_yolo_to_coco(arg.image_dir)
