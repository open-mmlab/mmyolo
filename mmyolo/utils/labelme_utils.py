# Copyright (c) OpenMMLab. All rights reserved.
import json

from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


class LabelmeFormat:
    """Predict results save into labelme file.

    Base on https://github.com/wkentaro/labelme/blob/main/labelme/label_file.py

    Args:
        classes (tuple): Model classes name.
        score_threshold (float): Predict score threshold.
    """

    def __init__(self, classes: tuple):
        super().__init__()
        self.classes = classes

    def __call__(self, results: DetDataSample, output_path: str,
                 pred_instances: InstanceData):
        """Get image data field for labelme.

        Args:
            results (DetDataSample): Predict info.
            output_path (str): Image file path.
            pred_instances (InstanceData): Candidate prediction info.

        Labelme file eg.
            {
              "version": "5.0.5",
              "flags": {},
              "imagePath": "/data/cat/1.jpg",
              "imageData": null,
              "imageHeight": 3000,
              "imageWidth": 4000,
              "shapes": [
                {
                  "label": "cat",
                  "points": [
                    [
                      1148.076923076923,
                      1188.4615384615383
                    ],
                    [
                      2471.1538461538457,
                      2176.923076923077
                    ]
                  ],
                  "group_id": null,
                  "shape_type": "rectangle",
                  "flags": {}
                },
                {...}
              ]
            }
        """

        image_path = results.metainfo['img_path']

        json_info = {
            'version': '5.0.5',
            'flags': {},
            'imagePath': image_path,
            'imageData': None,
            'imageHeight': results.ori_shape[0],
            'imageWidth': results.ori_shape[1],
            'shapes': []
        }

        for pred_info in pred_instances:
            pred_bbox = pred_info.bboxes.cpu().numpy().tolist()[0]
            pred_label = self.classes[pred_info.labels]

            sub_dict = {
                'label': pred_label,
                'points': [pred_bbox[:2], pred_bbox[2:]],
                'group_id': None,
                'shape_type': 'rectangle',
                'flags': {}
            }
            json_info['shapes'].append(sub_dict)

        with open(output_path, 'w', encoding='utf-8') as f_json:
            json.dump(json_info, f_json, ensure_ascii=False, indent=2)
