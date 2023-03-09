from .data_preprocessor import YOLOv5DetDataPreprocessor


class YOLO6DDetDataPreprocessor(YOLOv5DetDataPreprocessor):
    """Rewrite collate_fn to get faster training speed"""
    
    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion
        based on YOLOv5DetDataPreprocessor
        """
        
        if not training:
            # TODO: 推理阶段如何
            pass
        
        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        assert isinstance(data['data_samples'], dict)
        
        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std
        
        # TODO: not support
        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)
        
        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        # 
        data_samples = {
            'cpts_confs_labels': data_samples['cpts_confs_labels'],
            'img_metas': img_metas
        }
        
        return {'inputs': inputs, 'data_samples': data_samples} 