import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from mmcv.cnn import ConvModule
from ..builder import HEADS
from ..builder import build_loss
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
from .stdc_lk_attn_head import ShallowNet_lk_attn
from mmseg.models.losses.bce_dice_loss import BCEDiceLoss
from mmseg.models.losses.detail_loss import DetailAggregateLoss
from .isdhead import RelationAwareFusion
from mmseg.models.sampler.dysample import DySample
from other_utils.histogram import tensor_histogram
from mmseg.models.decode_heads.isdhead import SRDecoder
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from .stdc_rf import ShallowNet_rf63
from .stdc_rep_stem_head import ShallowNetRepStem





@HEADS.register_module()
class MyStdcHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels,img_size, reduce=False,decoder_flag='aff', **kwargs):
        entropy_topk_loss = kwargs.pop('entropy_topk_loss', None)
        super(MyStdcHead, self).__init__(**kwargs)
        self.decoder_flag=decoder_flag
        self.entropy_topk_loss = None
        if entropy_topk_loss is not None:
            entropy_topk_loss = entropy_topk_loss.copy()
            entropy_topk_loss_enabled = entropy_topk_loss.pop('enabled', False)
            if entropy_topk_loss_enabled:
                entropy_topk_loss.setdefault('type', 'EntropyTopKCrossEntropyLoss')
                self.entropy_topk_loss = build_loss(entropy_topk_loss)
        self.shallow_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
                type='BCEDiceLoss'))
        self.down_ratio = down_ratio
        self.convert_shallow8=nn.Conv2d(256,self.channels,stride=1,kernel_size=1)
        # self.stdc_net = ShallowNet_rf63(in_channels=3, pretrain_model="/root/autodl-tmp/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        # self.stdc_net = ShallowNet(in_channels=3, pretrain_model="/root/autodl-tmp/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        self.stdc_net = ShallowNet_lk(in_channels=3,num_classes=self.num_classes)
        # self.stdc_net = ShallowNet_lk_attn(in_channels=3,num_classes=self.num_classes)
        # self.stdc_net = ShallowNetRepStem(in_channels=3,num_classes=self.num_classes)
        # self.stdc_net = ShallowNet_rep(in_channels=3,num_classes=self.num_classes)
        # self.stdc_net = ShallowNet_rep(in_channels=3, pretrain_model="/root/autodl-tmp/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        self.reduce = Reducer() if reduce else None
        self.convert_shallow16=nn.Conv2d(512,self.channels,stride=1,kernel_size=3,padding=1)
        self.addConv=AddFuse(self.channels,self.channels)
        self.fuse=AddFuse(self.channels,self.channels)
        self.cls_seg8 = nn.Conv2d(128, 7, kernel_size=1)





    def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
        """Forward function."""
        # if train_flag==False:
        #     self.stdc_net.switch_to_deploy()
        # _, _, h, w = inputs.size()
        # inputs=F.interpolate(inputs, size=(h//2, w//2), mode='bilinear', align_corners=False)


        shallow_feat8, shallow_feat16 = self.stdc_net(inputs)

        # add fusion
        shallow_feat16 = self.convert_shallow16(shallow_feat16)
        shallow_feat8=self.convert_shallow8(shallow_feat8)
        _, _, h, w = shallow_feat8.size()
        shallow_feat16 = F.interpolate(shallow_feat16, size=(h , w ), mode='bilinear', align_corners=False)
        fusion=self.addConv(shallow_feat8,shallow_feat16)
        output = self.cls_seg(fusion)
        if train_flag==True:
            aux_output = self.cls_seg8(shallow_feat8)
            return output,aux_output
        else:
            return output







    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg,mask=None):
        output,aux_output = self.forward(
            inputs, prev_output,
            mask=mask,
            gt=gt_semantic_seg,
            img_metas=img_metas,
            train_cfg=train_cfg,
            diff_pred_deep=mask)
        losses = self.losses(output, gt_semantic_seg)
        if self.entropy_topk_loss is not None:
            entropy_topk_logit = resize(
                input=output,
                size=gt_semantic_seg.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            losses['loss_entropy_topk_ce'] = self.entropy_topk_loss(
                entropy_topk_logit,
                gt_semantic_seg,
                ignore_index=self.ignore_index)
        losses_aux = self.losses(aux_output, gt_semantic_seg)
        return  losses,losses_aux



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

