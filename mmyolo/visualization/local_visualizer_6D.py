from mmdet.visualization import DetLocalVisualizer
from mmyolo.registry import VISUALIZERS


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
        

        
    


    