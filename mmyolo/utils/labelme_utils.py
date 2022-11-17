# Copyright (c) OpenMMLab. All rights reserved.
import base64
import io
import json
import os

import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import torch
from mmdet.structures import DetDataSample


class LabelmeFormat:
    """Predict result save into labelme file.

    Base on https://github.com/wkentaro/labelme/blob/main/labelme/label_file.py
    """

    @staticmethod
    def get_image_exif_orientation(image) -> PIL.Image.Image:
        """Get image exif orientation info.

        Args:
            image (PIL Object): PIL Object.
        Return:
            (PIL Object): PIL image with correct orientation
        """
        try:
            exif = image.getexif()
        except AttributeError:
            return image

        image_exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in exif.items() if k in PIL.ExifTags.TAGS
        }

        orientation = image_exif.get('Orientation', None)

        if orientation == 1:
            # do nothing
            return image
        elif orientation == 2:
            # left-to-right mirror
            return PIL.ImageOps.mirror(image)
        elif orientation == 3:
            # rotate 180
            return image.transpose(PIL.Image.ROTATE_180)
        elif orientation == 4:
            # top-to-bottom mirror
            return PIL.ImageOps.flip(image)
        elif orientation == 5:
            # top-to-left mirror
            return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
        elif orientation == 6:
            # rotate 270
            return image.transpose(PIL.Image.ROTATE_270)
        elif orientation == 7:
            # top-to-right mirror
            return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
        elif orientation == 8:
            # rotate 90
            return image.transpose(PIL.Image.ROTATE_90)
        else:
            return

    def get_image_data(self, file_path: str) -> bytes:
        """Get image data field for labelme.

        Args:
            file_path (str): Image file path.
        Return:
            (str): Image file io bytes.
        """
        image_pil = PIL.Image.open(file_path)

        # get orientation to image according to exif
        image_pil = self.get_image_exif_orientation(image_pil)

        with io.BytesIO() as f:
            image_suffix = os.path.splitext(file_path)[1].lower()
            if image_suffix in ['.jpg', '.jpeg']:
                image_format = 'JPEG'
            else:
                image_format = 'PNG'
            image_pil.save(f, format=image_format)
            f.seek(0)
            return f.read()

    @staticmethod
    def img_data_to_pil(img_data):
        f = io.BytesIO()
        f.write(img_data)
        img_pil = PIL.Image.open(f)
        return img_pil

    def img_data_to_arr(self, img_data):
        img_pil = self.img_data_to_pil(img_data)
        img_arr = np.array(img_pil)
        return img_arr

    def img_b64_to_arr(self, img_b64):
        img_data = base64.b64decode(img_b64)
        img_arr = self.img_data_to_arr(img_data)
        return img_arr

    def get_image_height_and_width(self, image_data, image_height,
                                   image_width):
        img_arr = self.img_b64_to_arr(image_data)
        if image_height is not None and img_arr.shape[0] != image_height:
            print('imageHeight does not match with imageData or imagePath, '
                  'so getting imageHeight from actual image.')
            image_height = img_arr.shape[0]
        if image_width is not None and img_arr.shape[1] != image_width:
            print('imageWidth does not match with imageData or imagePath, '
                  'so getting imageWidth from actual image.')
            image_width = img_arr.shape[1]
        return image_height, image_width

    def __call__(self, pred_res: DetDataSample, output_path: str,
                 pred_score_thr: float, model_classes: tuple):
        """Get image data field for labelme.

        Args:
            pred_res (DetDataSample): Predict info.
            output_path (str): Image file path.
            pred_score_thr (float): Predict score threshold.
            model_classes (tuple): Model classes name.

        Labelme file eg.
            {
              "version": "5.0.5",
              "flags": {},
              "imagePath": "/data/cat/1.jpg",
              "imageData": "base64 data......",
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

        image_path = pred_res.metainfo['img_path']

        image_data = self.get_image_data(image_path)
        image_data = base64.b64encode(image_data).decode('utf-8')
        image_height, image_width = self.get_image_height_and_width(
            image_data, pred_res.ori_shape[0], pred_res.ori_shape[1])
        json_info = dict()
        json_info.update({
            'version': '5.0.5',
            'flags': {},
            'imagePath': image_path,
            'imageData': image_data,
            'imageHeight': image_height,
            'imageWidth': image_width,
            'shapes': []
        })

        res_index = torch.where(
            torch.tensor(pred_res.pred_instances.scores > pred_score_thr))

        for index in res_index:
            pred_bbox = pred_res.pred_instances.bboxes[index].cpu().numpy(
            ).tolist()[0]
            pred_label = model_classes[pred_res.pred_instances.labels[index]]

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
