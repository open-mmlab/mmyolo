import argparse
import os
import os.path as osp

import mmcv
import mmengine


def format_by_file(image_dir: str):
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
    assert osp.exists(image_dir)

    yolo_label_dir = osp.join(image_dir, 'labels')
    yolo_image_dir = osp.join(image_dir, 'images')
    yolo_class_txt = osp.join(image_dir, 'classes.txt')
    with open(yolo_class_txt) as f:
        classes = f.read().strip().split()

    indices = os.listdir(yolo_image_dir)

    if split:
        print(
            'Start to organize data according to existing train/test/val split'
        )
        train_dataset = {'images': [], 'annotations': [], 'categories': []}
        val_dataset = {'images': [], 'annotations': [], 'categories': []}
        test_dataset = {'images': [], 'annotations': [], 'categories': []}

        # categories id starts from 0
        for i, cls in enumerate(classes, 0):
            train_dataset['categories'].append({'id': i, 'name': cls})
            val_dataset['categories'].append({'id': i, 'name': cls})
            test_dataset['categories'].append({'id': i, 'name': cls})
        train_img, val_img, test_img = format_by_file(image_dir)
    else:
        print('Start to organize data')
        dataset = {'images': [], 'annotations': [], 'categories': []}
        for i, cls in enumerate(classes, 0):
            dataset['categories'].append({'id': i, 'name': cls})

    obj_count = 0
    for idx, file in enumerate(mmengine.track_iter_progress(indices)):
        # support both .jpg and .png
        txt_file = file.replace('images',
                               'txt').replace('.jpg',
                                              '.txt').replace('.png', '.txt')
        img = mmcv.imread(osp.join(image_dir, 'images/') + file)
        height, width = img.shape[:2]

        if split:
            if file in train_img:
                dataset = train_dataset
            elif file in val_img:
                dataset = val_dataset
            elif file in test_img:
                dataset = test_dataset

        dataset['images'].append({
            'file_name': file,
            'id': idx,
            'width': width,
            'height': height
        })

        if not osp.exists(osp.join(yolo_label_dir, txt_file)):
            # if current image is not annotated
            continue

        with open(osp.join(yolo_label_dir, txt_file)) as f:
            labels = f.readlines()
            for label in labels:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W = img.shape[:2]
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H

                cls_id = int(label[0])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'image_id':
                    idx,
                    'id':
                    obj_count,
                    'category_id':
                    cls_id,
                    'bbox': [x1, y1, width, height],
                    'area':
                    width * height,
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
                    'iscrowd':
                    0
                })
                obj_count += 1

    # saving results
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
    print('Finished')


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
