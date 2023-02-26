import numpy as np
from typing import Optional, Union, List

import torch
import mmcv
from mmyolo.structures import DataSample6D
from mmdet.visualization import DetLocalVisualizer
from mmyolo.registry import VISUALIZERS
from mmengine.dist import master_only
from mmdet.visualization import get_palette
from mmengine.visualization.utils import check_type, tensor2ndarray

def _get_adaptive_scales(areas: np.ndarray,
                         min_area: int = 800,
                         max_area: int = 30000) -> np.ndarray:
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``min_area``, the scale is 0.5 while the area is larger than
    ``max_area``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Defaults to 800.
        max_area (int): Upper bound areas for adaptive scales.
            Defaults to 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales

def remove_prefix(key: str) -> str:
    if key.startswith('gt_'):
        key = key[3:]
    if key.startswith('pred_'):
        key = key[5:]
    return key

@VISUALIZERS.register_module()
class LocalVisualizer6D(DetLocalVisualizer):
    """6D Local Visualizer"""
        
    def _draw_instances(self,
                        image,
                        instances,
                        classes,
                        palette):

        self.set_image(image)
        
        if 'bboxes' in instances:
            bboxes = instances.bboxes
            labels = instances.labels
            
            max_label = int(max(labels)) if len(labels)>0 else 0
            
            text_palette = get_palette(self.text_color, max_label+1)
            text_colors = [text_palette[label] for label in labels]
            
            bbox_color = palette if self.bbox_color is None\
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label+1)
            
            colors = [bbox_palette[label] for label in labels]
            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha = self.alpha,
                line_widths = self.line_width
            )
            
            positions = bboxes[:, :2] + self.line_width
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)
            
            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'
                if 'scores' in instances:
                    score = round(float(instances.scores[i])*100, 1)
                    label_text += f': {score}'
                
                self.draw_texts(
                    label_text,
                    pos,
                    colors=text_colors[i],
                    font_sizes=int(13*scales[i]),
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }]
                )
                
        if 'translation' in instances:
            center = instances.center
            corners = instances.corners
            
            max_label = int(max(labels) if len(labels)>0 else 0)
            
            bbox_color_6d = palette if self.bbox_color is None\
                else self.bbox_color
            bbox_palette_6d = get_palette(bbox_color_6d, max_label+1)
            colors = [bbox_palette_6d[label] for label in labels]
            
            for i in range(len(corners)):
                self.draw_bboxes_6d(
                    corners[i],
                    edge_colors=colors,
                    line_widths=self.line_width
                )
            for i in range(len(center)):
                self.draw_points(center[i],
                                 colors='r')
                
        return self.get_image()
            
    
    def draw_bboxes_6d(
        self,
        corners: Union[np.ndarray, torch.Tensor],
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'r',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
    ):
        check_type('corners', corners, (np.ndarray, torch.Tensor))
        corners = tensor2ndarray(corners)
        edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3],
                         [1, 5], [2, 3], [2, 6], [3, 7],
                         [4, 5], [4, 6], [5, 7], [6, 7]]

        if len(corners.shape) == 1:
            corners = corners[None]
        assert corners.shape[-1] == 16, (
            f'The shape of `bbox_6d` should be (N, 16), but got {corners.shape}')
        corners = corners.reshape(8, 2)
        for edge in edges_corners:
            self.draw_lines(corners[edge, 0],
                            corners[edge, 1],
                            colors=edge_colors, 
                            line_styles=line_styles,
                            line_widths=line_widths)
        return self
            
    @master_only
    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       draw_gt: bool = True,
                       data_sample: Optional['DataSample6D'] = None,
                       draw_pred: bool = False,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       pred_score_thr: float = 0.3,
                       save_path: Optional[str] = None,
                       step: int = 0) -> None:
        classes = self.dataset_meta.get('CLASSES', None)
        palette = self.dataset_meta.get('PALETTE', None)
        
        gt_img_data = None
        pred_img_data = None
        
        if data_sample is not None:
            data_sample = data_sample.cpu()
        
        if draw_gt and data_sample is not None:
            gt_img_data = image
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances(image,
                                                   data_sample.gt_instances,
                                                   classes,
                                                   palette)
        
        if draw_pred and data_sample is not None:
            pred_img_data = image
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores>pred_score_thr]
                pred_img_data = self._draw_instances(image,
                                                     pred_instances,
                                                     classes,
                                                     palette)
        
        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            drawn_img = image
        
        self.set_image(drawn_img)
        
        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)
        
        if out_file is not None:
            mmcv.imwrite(drawn_img[...,::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)

