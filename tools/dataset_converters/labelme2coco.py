import argparse
import json
from pathlib import Path

import numpy as np

from mmyolo.utils.misc import IMG_EXTENSIONS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, help='Dataset image directory')
    parser.add_argument('--labels-dir', type=str, help='Dataset labels directory')
    parser.add_argument('--out-path', type=str, help='COCO label json output path')
    args = parser.parse_args()
    return args


def gen_coco_annotations_rectangle(points, image_id, annotations_id, category_id) -> dict:
    """Gen COCO annotations format label from labelme format label.

    Args:
        points (list): Image dir path.
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
    annotation_info['bbox'] = list(
        map(float, [
            points[0][0],
            points[0][1],
            points[1][0] - points[0][0],
            points[1][1] - points[0][1]
        ]))
    annotation_info['area'] = annotation_info['bbox'][2] * annotation_info['bbox'][3]
    segmentation_points = np.asarray(points).copy()
    segmentation_points[1, :] = np.asarray(points)[2, :]
    segmentation_points[2, :] = np.asarray(points)[1, :]
    annotation_info['segmentation'] = [list(segmentation_points.flatten())]

    return annotation_info


def gen_coco_json(image_dir, labels_root):
    """Gen COCO json format label from labelme format label.

    Args:
        image_dir (str): Image dir path.
        labels_root (str): Image label root path.

    Return:
        coco_json (dict): COCO json data.
    """

    # init coco json field
    coco_json = {
        'images': [],
        'categories': [],
        'annotations': []
    }

    image_id = 0
    annotations_id = 0
    category_to_id = dict()
    categories_labels = []

    for img_file in Path(image_dir).iterdir():
        # filter incorrect image file
        if img_file.suffix.lower() not in IMG_EXTENSIONS:
            continue

        # get label file according to the image file name
        label_path = Path(labels_root).joinpath(img_file.stem).with_suffix('.json')
        if not label_path.exists():
            print(f"Can not find {label_path}, skip...")
            continue

        # load labelme label
        with open(label_path, 'r', encoding='utf-8') as f:
            print(f'Convert labelme to COCO from: {label_path}')
            labelme_data = json.load(f)

        image_id = image_id + 1  # coco id begin from 1

        # update coco 'images' field
        coco_json['images'].append({
            'height': labelme_data['imageHeight'],
            'width': labelme_data['imageWidth'],
            'id': image_id,
            'file_name': Path(labelme_data['imagePath']).name
        })

        for label_shapes in labelme_data['shapes']:

            # Update coco 'categories' field
            class_name = label_shapes['label']
            if class_name not in categories_labels:
                # only update when not been added before
                coco_json['categories'].append({
                    'supercategory': 'component',
                    'id': len(categories_labels) + 1,
                    'name': class_name
                })
                categories_labels = categories_labels.append(class_name)
                category_to_id[class_name] = len(categories_labels)

            # get shape type and convert it to coco format
            shape_type = label_shapes['shape_type']
            if shape_type != 'rectangle':
                print(f'not support `{shape_type}` yet, skip...')
                continue

            annotations_id = annotations_id + 1
            # convert point from [xmin, ymin, xmax, ymax] to [x1, y1, w, h]
            (x1, y1), (x2, y2) = label_shapes['points']
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            points = [[x1, y1], [x2, y2], [x1, y2], [x2, y1]]
            coco_json['annotations'].append(
                gen_coco_annotations_rectangle(points, image_id, annotations_id, category_to_id[class_name]))

    return coco_json


def convert_labelme_to_coco(image_dir: str, labels_dir: str, out_path: str):
    """Convert labelme format label to COCO json format label.

    Args:
        image_dir (str): Image dir path.
        labels_dir (str): Image label path.
        out_path (str): COCO json file save path.

    """
    assert Path(out_path).suffix == '.json'

    # convert to coco json
    coco_json_data = gen_coco_json(image_dir, labels_dir)

    # save json result
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving json to {out_path}")
    json.dump(coco_json_data, open(out_path, 'w'), indent=4)


def main():
    args = parse_args()
    convert_labelme_to_coco(args.images_dir, args.labels_dir, args.out_path)
    print("All done!")


if __name__ == '__main__':
    main()
