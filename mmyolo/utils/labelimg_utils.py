import io
import json
import os

import numpy as np
import torch
from mmdet.structures import DetDataSample

class LabelimgFormat:
  """Predict results save into labelimg file.
  
  Args:
     classes (tuple): Model classes name.
     
 """
 def __init__(self, classes: tuple):
  super().__init__()
  self.classes = classes
  
 def get_image_exif_orientation(image) -> Image.Image:
        """Get image exif orientation info.
        Args:
            image (PIL Object): PIL Object.
        Return:
            (PIL Object): PIL image with correct orientation
        """
