from ..builder import HEADS
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from .refine_decode_head import RefineBaseDecodeHead



@HEADS.register_module()
class SwinUnetHead(RefineBaseDecodeHead):
    def __init__(self,**kwargs):
        super(SwinUnetHead, self).__init__(**kwargs)
        self.change=nn.Conv2d(in_channels=self.in_channels,out_channels=self.channels,stride=2,kernel_size=1)
        self.output = nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_classes, kernel_size=1, bias=False)

    def forward(self,input):
        fm_middle=[]
        t=self.change(input)
        fm_middle.append(t)
        output=self.output(input)
        fm_middle.append(output)
        return output, fm_middle
