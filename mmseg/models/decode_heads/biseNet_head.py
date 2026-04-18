import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from .diff_fusion import FeatureFusionModule
from .harr import HarrUp
from other_utils.heatmap import save_image,save_heatmap,visualize_feature_map
from mmseg.ops import resize
from mmseg.models.losses.detail_loss import DetailAggregateLoss
from .pid import Bag,AdaptiveFrequencyFusion,AddFuse
from .diff_head import DiffHead
from .diff_point import DiffPoint
from .spnet import SpNet,get_coordsandfeatures,SparseResNet50
from .spnetv2 import SpNetV2
from other_utils.split_tensor import split_tensor,restore_tensor
from .pid import segmenthead,CBAMLayer
from .sdd_stdc_head import ShallowNet
from .stdc_rep_head import ShallowNet_rep
from .stdc_lk_head import ShallowNet_lk
from mmseg.models.losses.bce_dice_loss import BCEDiceLoss
from mmseg.models.losses.detail_loss import DetailAggregateLoss
from .isdhead import RelationAwareFusion
from mmseg.models.sampler.dysample import DySample
from other_utils.histogram import tensor_histogram
from mmseg.models.decode_heads.isdhead import SRDecoder
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from .bisenet import SP,CP,BiSeNet
from .rep_resnet import RepSP
from .resnet_ks import SPNoDownsample
from .rep_dilated_resnet import ParallelDilatedSP





@HEADS.register_module()
class BiseNetHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels,img_size, reduce=False,decoder_flag='aff', **kwargs):
        super(BiseNetHead, self).__init__(**kwargs)
        self.decoder_flag=decoder_flag
        self.shallow_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
                type='BCEDiceLoss'))
        self.down_ratio = down_ratio
        self.convert_shallow8=nn.Conv2d(256,self.channels,stride=1,kernel_size=1)
        # self.bise_net=SP()
        # self.bise_net=RepSP()
        # self.bise_net=SPNoDownsample()
        self.bise_net=ParallelDilatedSP()
        # self.bise_net=CP()
        # self.bise_net=BiseNetHead(num_classes=self.num_classes)
        # self.stdc_net = ShallowNet(in_channels=3, pretrain_model="/root/autodl-tmp/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        # self.stdc_net = ShallowNet_lk(in_channels=3,num_classes=self.num_classes)
        # self.stdc_net = ShallowNet_rep(in_channels=3, pretrain_model="/root/autodl-tmp/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        self.reduce = Reducer() if reduce else None
        self.convert_shallow16=nn.Conv2d(512,self.channels,stride=1,kernel_size=3,padding=1)
        self.addConv=AddFuse(self.channels,self.channels)
        self.fuse=AddFuse(self.channels,self.channels)
        self.cls_seg8 = nn.Conv2d(128, 7, kernel_size=1)



    # def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
    #     """Forward function."""
    #     feat = self.bise_net(inputs)
    #     output = self.cls_seg(feat)
    #     return output


    def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
        """Forward function."""
        inputs = nn.functional.interpolate(inputs, size=[inputs.shape[-2] // 4,
                                                       inputs.shape[-1] // 4])
        feat = self.bise_net(inputs)
        output = self.cls_seg(feat)
        return output








    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg,mask=None):
        output = self.forward(
            inputs, prev_output,
            mask=mask,
            gt=gt_semantic_seg,
            img_metas=img_metas,
            train_cfg=train_cfg,
            diff_pred_deep=mask)
        losses = self.losses(output, gt_semantic_seg)
        return  losses



    def forward_test(self, inputs, prev_output, img_metas, test_cfg,mask=None):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """

        return self.forward(inputs, prev_output, False,diff_pred_deep=mask)

