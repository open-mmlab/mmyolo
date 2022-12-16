import math

import mmcv
import numpy as np
import torch
from mmdet.visualization import DetLocalVisualizer
from mmdet.visualization.palette import _get_adaptive_scales, get_palette
from mmengine.visualization import Visualizer
from torch import Tensor

from mmyolo.registry import VISUALIZERS


@VISUALIZERS.register_module()
class DetAssignerVisualizer(DetLocalVisualizer):
    # def __init__(self, *args, **kwargs):
    #     pass
    #     super().__init__(*args, **kwargs)
    #     self.priors_size = None

    def draw_grid(
            self,
            stride=8,
            line_styles=':',
            colors=(180, 180, 180),
            line_widths=1
    ):
        assert self._image is not None, 'Please set image using `set_image`'
        # draw vertical lines
        x_datas_vertical = ((np.arange(self.width // stride - 1) + 1) * stride).reshape((-1, 1)).repeat(2, axis=1)
        y_datas_vertical = np.array([[0, self.height - 1]]).repeat(self.width // stride - 1, axis=0)
        self.draw_lines(x_datas_vertical, y_datas_vertical, colors=colors, line_styles=line_styles,
                        line_widths=line_widths)

        # draw horizontal lines
        x_datas_horizontal = np.array([[0, self.width - 1]]).repeat(self.height // stride - 1, axis=0)
        y_datas_horizontal = ((np.arange(self.height // stride - 1) + 1) * stride).reshape((-1, 1)).repeat(2, axis=1)
        self.draw_lines(x_datas_horizontal, y_datas_horizontal, colors=colors, line_styles=line_styles,
                        line_widths=line_widths)

        return self

    def draw_instances_assign(
            self,
            instances,
            retained_gt_inds,
            not_show_label=False):
        assert self.dataset_meta is not None
        classes = self.dataset_meta['CLASSES']
        palette = self.dataset_meta['PALETTE']
        if len(retained_gt_inds) == 0:
            return self.get_image()
        draw_gt_inds = torch.from_numpy(np.array(list(set(retained_gt_inds.cpu().numpy())), dtype=np.int64))
        bboxes = instances.bboxes[draw_gt_inds]
        labels = instances.labels[draw_gt_inds]

        if not isinstance(bboxes, Tensor):
            bboxes = bboxes.tensor

        edge_colors = [palette[i] for i in labels]

        max_label = int(max(labels) if len(labels) > 0 else 0)
        text_palette = get_palette(self.text_color, max_label + 1)
        text_colors = [text_palette[label] for label in labels]

        self.draw_bboxes(bboxes, edge_colors=edge_colors,
                         alpha=self.alpha, line_widths=self.line_width)

        if not not_show_label:
            positions = bboxes[:, :2] + self.line_width
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                    bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)
            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'

                self.draw_texts(
                    label_text,
                    pos,
                    colors=text_colors[i],
                    font_sizes=int(13 * scales[i]),
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])
        return self.get_image()

    def draw_positive_assign(self, grid_x_inds, grid_y_inds, class_inds, stride, bboxes, retained_gt_inds, offset=0.5):
        if not isinstance(bboxes, Tensor):
            bboxes = bboxes.tensor

        assert self.dataset_meta is not None
        palette = self.dataset_meta['PALETTE']
        x = ((grid_x_inds + offset) * stride).long()
        y = ((grid_y_inds + offset) * stride).long()
        center = torch.stack((x, y), dim=-1)
        # radius = torch.ones_like(x) * (stride // 4)

        retained_bboxes = bboxes[retained_gt_inds]
        bbox_wh = retained_bboxes[:, 2:] - retained_bboxes[:, :2]
        bbox_area = bbox_wh[:, 0] * bbox_wh[:, 1]
        radius = _get_adaptive_scales(bbox_area) * 4
        colors = [palette[i] for i in class_inds]

        self.draw_circles(center, radius, colors, line_widths=0,
                          face_colors=colors, alpha=1.0)

    def draw_prior(self, grid_x_inds, grid_y_inds, stride, class_inds, ind_feat, ind_prior, offset=0.5):

        palette = self.dataset_meta['PALETTE']
        center_x = ((grid_x_inds + offset) * stride)
        center_y = ((grid_y_inds + offset) * stride)
        xyxy = torch.stack((center_x, center_y, center_x, center_y), dim=1)
        xyxy += self.priors_size[ind_feat][ind_prior]

        colors = [palette[i] for i in class_inds]
        self.draw_bboxes(xyxy, edge_colors=colors,
                         alpha=self.alpha,
                         line_styles='--',
                         line_widths=math.ceil(self.line_width * 0.3))

    def draw_assign(self, img, assign_results, gt_instances, show_prior=False, not_show_label=False):
        img_show_list = []
        for ind_feat, assign_results_feat in enumerate(assign_results):
            img_show_list_feat = []
            for ind_prior, assign_results_prior in enumerate(assign_results_feat):
                self.set_image(img)
                h, w = img.shape[:2]

                # draw grid
                stride = assign_results_prior['stride']
                self.draw_grid(stride)

                # draw prior on matched gt
                grid_x_inds = assign_results_prior['grid_x_inds']
                grid_y_inds = assign_results_prior['grid_y_inds']
                class_inds = assign_results_prior['class_inds']
                ind_prior = assign_results_prior['prior_ind']
                if show_prior:
                    self.draw_prior(grid_x_inds, grid_y_inds, stride, class_inds, ind_feat, ind_prior)

                # draw matched gt
                retained_gt_inds = assign_results_prior['retained_gt_inds']
                self.draw_instances_assign(gt_instances, retained_gt_inds, not_show_label)

                # draw positive
                self.draw_positive_assign(grid_x_inds, grid_y_inds, class_inds, stride, gt_instances.bboxes,
                                          retained_gt_inds)

                # draw title
                base_prior = self.priors_size[ind_feat][ind_prior]
                prior_size = (base_prior[2] - base_prior[0], base_prior[3] - base_prior[1])
                pos = np.array((20, 20))
                text = f'feat_ind: {ind_feat}  prior_ind: {ind_prior} prior_size: ({prior_size[0]}, {prior_size[1]})'
                scales = _get_adaptive_scales(np.array([h * w / 16]))
                font_sizes = int(13 * scales)
                self.draw_texts(
                    text,
                    pos,
                    colors=self.text_color,
                    font_sizes=font_sizes,
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

                img_show = self.get_image()
                img_show = mmcv.impad(img_show, padding=(5, 5, 5, 5))
                img_show_list_feat.append(img_show)
            img_show_list.append(np.concatenate(img_show_list_feat, axis=1))
        return np.concatenate(img_show_list, axis=0)
