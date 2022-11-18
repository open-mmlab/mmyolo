# Copyright (c) OpenMMLab. All rights reserved.
import json

import torch
from mmdet.structures import DetDataSample


class LabelmeFormat:
    """Predict results save into labelme file.

    Base on https://github.com/wkentaro/labelme/blob/main/labelme/label_file.py

    Args:
        classes (tuple): Model classes name.
        score_threshold (float): Predict score threshold.
    """

    def __init__(self, classes: tuple, score_threshold: float):
        super().__init__()
        self.classes = classes
        self.score_threshold = score_threshold

    def __call__(self, results: DetDataSample, output_path: str):
        """Get image data field for labelme.

        Args:
            results (DetDataSample): Predict info.
            output_path (str): Image file path.

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

        res_index = torch.where(
            torch.tensor(
                results.pred_instances.scores > self.score_threshold))[0]

        for index in res_index:
            pred_bbox = results.pred_instances.bboxes[index].cpu().numpy(
            ).tolist()
            pred_label = self.classes[results.pred_instances.labels[index]]

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
