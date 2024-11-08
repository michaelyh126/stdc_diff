import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .refine_decode_head import RefineBaseDecodeHead




@HEADS.register_module()
class UncertaintyHead(RefineBaseDecodeHead):
    def __init__(self,**kwargs):
        super(UncertaintyHead, self).__init__(**kwargs)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.in_channels),
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(self.channels, 1)
        )

    def forward(self,input):
        uncertainty_num=self.conv_layers(input)
        uncertainty_num=self.global_pool(uncertainty_num)
        uncertainty_num = torch.flatten(uncertainty_num, start_dim=1)
        uncertainty_num=self.fc(uncertainty_num)
        return uncertainty_num
