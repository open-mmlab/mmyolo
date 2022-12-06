from mmdet.visualization import DetLocalVisualizer

from mmyolo.registry import VISUALIZERS



@VISUALIZERS.register_module()
class AssignerVisualizer(DetLocalVisualizer):
    def _draw_gt(self):
        pass

    def _draw_anchor(self):
        pass

    def _draw_assign_result(self):
        pass
