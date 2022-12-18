# Copyright (c) OpenMMLab. All rights reserved.
"""This script helps to convert visdrone-style dataset to the coco format.

Usage:
    $ python visdrone2coco.py \
                --img-dir /path/to/images \
                --labels-dir /path/to/labels \
                --out /path/to/coco_instances.json \

"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from mmengine import track_iter_progress
from mmyolo.utils.misc import IMG_EXTENSIONS
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, help='Dataset image directory')
    parser.add_argument(
        '--labels-dir', type=str, help='Dataset labels directory')
    parser.add_argument('--out', type=str, help='COCO label json output path')
    args = parser.parse_args()
    return args


def format_coco_annotations(points: list, image_id: int, annotations_id: int,
                            category_id: int) -> dict:
    """Gen COCO annotations format label from labelme format label.
    Args:
        points (list): Coordinates of four vertices of rectangle bbox.
        image_id (int): Image id.
        annotations_id (int): Annotations id.
        category_id (int): Image dir path.
    Return:
        annotation_info (dict): COCO annotation data.
    """
    annotation_info = dict()
    annotation_info['iscrowd'] = 0
    annotation_info['category_id'] = category_id
    annotation_info['id'] = annotations_id
    annotation_info['image_id'] = image_id

    # bbox is [x1, y1, w, h]
    annotation_info['bbox'] = [
        points[0][0], points[0][1], points[1][0] - points[0][0],
        points[1][1] - points[0][1]
    ]

    annotation_info['area'] = annotation_info['bbox'][2] * annotation_info[
        'bbox'][3]  # bbox w * h
    segmentation_points = np.asarray(points).copy()
    segmentation_points[1, :] = np.asarray(points)[2, :]
    segmentation_points[2, :] = np.asarray(points)[1, :]
    annotation_info['segmentation'] = [list(segmentation_points.flatten())]

    return annotation_info


def parse_labelme_to_coco(
        image_dir: str,
        labels_root: str) -> dict:
    """Gen COCO json format label from labelme format label.
    Args:
        image_dir (str): Image dir path.
        labels_root (str): Image label root path.
    Return:
        coco_json (dict): COCO json data.
    COCO json example:
    {
        "images": [
            {
                "height": 3000,
                "width": 4000,
                "id": 1,
                "file_name": "IMG_20210627_225110.jpg"
            },
            ...
        ],
        "categories": [
            {
                "id": 1,
                "name": "cat"
            },
            ...
        ],
        "annotations": [
            {
                "iscrowd": 0,
                "category_id": 1,
                "id": 1,
                "image_id": 1,
                "bbox": [
                    1183.7313232421875,
                    1230.0509033203125,
                    1270.9998779296875,
                    927.0848388671875
                ],
                "area": 1178324.7170306593,
                "segmentation": [
                    [
                        1183.7313232421875,
                        1230.0509033203125,
                        1183.7313232421875,
                        2157.1357421875,
                        2454.731201171875,
                        2157.1357421875,
                        2454.731201171875,
                        1230.0509033203125
                    ]
                ]
            },
            ...
        ]
    }
    """

    # init coco json field
    coco_json = {'images': [], 'categories': [], 'annotations': []}

    image_id = 0
    annotations_id = 0
    all_classes_id = [
        {"id": 0, "name": "ignored regions"},
        {"id": 1, "name": "pedestrian"},
        {"id": 2, "name": "people"},
        {"id": 3, "name": "bicycle"},
        {"id": 4, "name": "car"},
        {"id": 5, "name": "van"},
        {"id": 6, "name": "truck"},
        {"id": 7, "name": "tricycle"},
        {"id": 8, "name": "awning-tricycle"},
        {"id": 9, "name": "bus"},
        {"id": 10, "name": "motor"},
        {"id": 11, "name": "others"}
    ]
    # filter incorrect image file
    img_file_list = [
        img_file for img_file in Path(image_dir).iterdir()
        if img_file.suffix.lower() in IMG_EXTENSIONS
    ]

    for img_file in track_iter_progress(img_file_list):
        # get label file according to the image file name
        label_path = Path(labels_root).joinpath(
            img_file.stem).with_suffix('.txt')
        if not label_path.exists():
            print(f'Can not find label file: {label_path}, skip...')
            continue
        
        # load image meta_data
        height, width = cv2.imread(img_file).shape[:2]
        image_id = image_id + 1  # coco id begin from 1

        # update coco 'images' field
        coco_json['images'].append({
            'height':
            height,
            'width':
            width,
            'id':
            image_id,
            'file_name':
            Path(img_file).name
        })

        # load labelme label
        with open(label_path, encoding='utf-8') as f:
            for line in f.readlines():
                annotations_id = annotations_id + 1
                line = line.replace("\n", "")
                if line.endswith(","):  # filter data
                    line = line.rstrip(",")
                    line_list = [int(i) for i in line.split(",")]
                    bbox_xywh = [line_list[0], line_list[1], line_list[2], line_list[3]]
                    class_id = int(line_list[5])
                    x1, y1, x2, y2 = bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]
                    x1, x2 = sorted([x1, x2])  # xmin, xmax
                    y1, y2 = sorted([y1, y2])  # ymin, ymax
                    points = [[x1, y1], [x2, y2], [x1, y2], [x2, y1]]
                    coco_annotations = format_coco_annotations(
                        points, image_id, annotations_id, class_id)
            coco_json['annotations'].append(coco_annotations)

    coco_json["categories"] = all_classes_id
    print(f'Total image = {image_id}')
    print(f'Total annotations = {annotations_id}')

    return coco_json
