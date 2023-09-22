# Copyright (c) OpenMMLab. All rights reserved.
from clip.model import ModifiedResNet
from torch.nn.modules.batchnorm import _BatchNorm

from mmyolo.registry import MODELS


@MODELS.register_module()
class CLIPModifiedResNet(ModifiedResNet):

    def __init__(self, freeze_backbone, freeze_bn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_backbone = freeze_backbone
        self.freeze_bn = freeze_bn

    def forward(self, x):

        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        # x = self.attnpool(x)
        return [x1, x2, x3]

    def train(self, mode=True):
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        elif self.freeze_backbone:  # 写法要优化
            for ind, m in enumerate(self.modules()):
                if ind != 0:
                    m.eval()
