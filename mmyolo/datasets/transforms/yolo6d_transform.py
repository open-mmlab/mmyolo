import os
import random
import numpy as np
import mmcv
import cv2
from PIL import Image, ImageMath
from mmyolo.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

def change_background(img, mask, bg):
        ow, oh = img.size
        bg = bg.resize((ow, oh)).convert('RGB')
        
        imcs = list(img.split())
        bgcs = list(bg.split())
        maskcs = list(mask.split())
        fics = list(Image.new(img.mode, img.size).split())
        
        for c in range(len(imcs)):
            negmask = maskcs[c].point(lambda i: 1 - i / 255)
            posmask = maskcs[c].point(lambda i: i / 255)
            fics[c] = ImageMath.eval("a * c + b * d", a=imcs[c], b=bgcs[c], c=posmask, d=negmask).convert('L')
        out = Image.merge(img.mode, tuple(fics))
        
        return out


@TRANSFORMS.register_module()
class CopyPaste6D(BaseTransform):
    """change the background"""
    def __init__(self,
                 shape,
                 num_keypoints,
                 max_num_gt
    ):
        self.shape = shape
        self.num_keypoints = num_keypoints
        self.max_num_gt = max_num_gt

    def transform(self, results:dict) -> dict:
        ## data augmentation
        img = results['img']
        maskpath = results['mask_path']
        bgpath = results['bg_path']
        
        # opencv -> PIL 
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask = Image.open(maskpath)
        bg = Image.open(bgpath)
        
        img = change_background(img, mask, bg)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        
        # match shape
        if self.shape is not None:
            oh, ow = img.shape[:2]
            sx = self.shape[0]/ow
            sy = self.shape[1]/oh
            scale_ratio = max(sx, sy)

            img = mmcv.imresize(img, (int(ow*scale_ratio),
                                      int(oh*scale_ratio)))
            img = img[0:self.shape[1], 0:self.shape[0]]
            
            results['gt_cpts_norm'][..., 0] *= scale_ratio
            results['gt_cpts_norm'][..., 1] *= scale_ratio
            
            results['gt_bboxes'] *= scale_ratio

        # PIL -> opencv
        results['img'] = img
        
        return results