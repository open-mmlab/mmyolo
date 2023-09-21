import json
import os

import cv2
from tqdm import tqdm

# Recommended dataset download address:
# https://cdn.opendatalab.com/Visdrone_DET/raw/Visdrone_DET.tar.gz


def convert_visdrone_to_coco(ann_file, out_file, image_prefix):
    id_num = 0
    img_num = 0
    categories = [{
        'id': 0,
        'name': 'ignored regions'
    }, {
        'id': 1,
        'name': 'pedestrian'
    }, {
        'id': 2,
        'name': 'people'
    }, {
        'id': 3,
        'name': 'bicycle'
    }, {
        'id': 4,
        'name': 'car'
    }, {
        'id': 5,
        'name': 'van'
    }, {
        'id': 6,
        'name': 'truck'
    }, {
        'id': 7,
        'name': 'tricycle'
    }, {
        'id': 8,
        'name': 'awning-tricycle'
    }, {
        'id': 9,
        'name': 'bus'
    }, {
        'id': 10,
        'name': 'motor'
    }, {
        'id': 11,
        'name': 'others'
    }]
    images = []
    annotations = []
    print('start loading data')
    set = os.listdir(ann_file)
    annotations_path = ann_file
    images_path = image_prefix
    for i in tqdm(set):
        f = open(annotations_path + '/' + i)
        name = i.replace('.txt', '')
        image = {}
        height, width = cv2.imread(images_path + '/' + name + '.jpg').shape[:2]
        file_name = name + '.jpg'
        image['file_name'] = file_name
        image['height'] = height
        image['width'] = width
        image['id'] = img_num
        images.append(image)
        for line in f.readlines():
            annotation = {}
            line = line.replace('\n', '')
            if line.endswith(','):
                line = line.rstrip(',')
            line_list = [int(i) for i in line.split(',')]
            bbox_xywh = [
                line_list[0], line_list[1], line_list[2], line_list[3]
            ]
            annotation['image_id'] = img_num
            annotation['score'] = line_list[4]
            annotation['bbox'] = bbox_xywh
            annotation['category_id'] = int(line_list[5])
            annotation['id'] = id_num
            annotation['iscrowd'] = 0
            annotation['segmentation'] = []
            annotation['area'] = bbox_xywh[2] * bbox_xywh[3]
            id_num += 1
            annotations.append(annotation)
        img_num += 1
    dataset_dict = {}
    dataset_dict['images'] = images
    dataset_dict['annotations'] = annotations
    dataset_dict['categories'] = categories
    json_str = json.dumps(dataset_dict)
    with open(out_file, 'w') as json_file:
        json_file.write(json_str)


print('json file write done')

if __name__ == '__main__':
    convert_visdrone_to_coco(
        '../data/visdrone/VisDrone2019-DET-train/annotations',
        '../data/visdrone/train_coco.json',
        '../data/visdrone/VisDrone2019-DET-train/images')
    convert_visdrone_to_coco(
        '../data/visdrone/VisDrone2019-DET-val/annotations',
        '../data/visdrone/val_coco.json',
        '../data/visdrone/VisDrone2019-DET-val/images')
