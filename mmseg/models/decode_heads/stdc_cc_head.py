import torch
import math
import os
import time
import torch.nn as nn
import torch.nn.functional as F
# import cv2
import numpy as np
from torch.nn import Softmax
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
from .stdc_lk_head import ShallowNet_lk,ShallowNet_lk2
from mmseg.models.losses.bce_dice_loss import BCEDiceLoss
from mmseg.models.losses.detail_loss import DetailAggregateLoss
from .isdhead import RelationAwareFusion
from mmseg.models.sampler.dysample import DySample
from other_utils.histogram import tensor_histogram
from mmseg.models.decode_heads.isdhead import SRDecoder
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from .stdc_rf import ShallowNet_rf63
from .pidnet_single import PIDNet
from inplace_abn import InPlaceABN, InPlaceABNSync


def split_to_patches(inputs, patch_size):
    """
    inputs: [B, C, H, W]
    patch_size: x

    return:
        patches: [B * n_h * n_w, C, x, x]
        meta: dict, 用于恢复
    """
    B, C, H, W = inputs.shape
    x = patch_size

    assert H % x == 0 and W % x == 0, \
        f"H and W must be divisible by patch_size, got H={H}, W={W}, patch_size={x}"

    n_h = H // x
    n_w = W // x

    # [B, C, H, W] -> [B, C, n_h, x, n_w, x]
    patches = inputs.reshape(B, C, n_h, x, n_w, x)

    # -> [B, n_h, n_w, C, x, x]
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()

    # -> [B*n_h*n_w, C, x, x]
    patches = patches.reshape(B * n_h * n_w, C, x, x)

    meta = {
        'B': B,
        'H': H,
        'W': W,
        'patch_size': x,
        'n_h': n_h,
        'n_w': n_w,
    }

    return patches, meta


def merge_patches(outputs, meta, out_size=None):
    """
    outputs: [B * n_h * n_w, C, h_p, w_p]
    meta: split_to_patches 返回的 meta
    out_size: 最终想恢复到的尺寸 (H, W)，默认用原图尺寸

    return:
        merged: [B, C, H_out, W_out]
    """
    B = meta['B']
    n_h = meta['n_h']
    n_w = meta['n_w']
    H = meta['H']
    W = meta['W']

    BN, C, h_p, w_p = outputs.shape
    assert BN == B * n_h * n_w, \
        f"Batch size mismatch: got {BN}, expected {B*n_h*n_w}"

    # [B*n_h*n_w, C, h_p, w_p]
    # -> [B, n_h, n_w, C, h_p, w_p]
    outputs = outputs.reshape(B, n_h, n_w, C, h_p, w_p)

    # -> [B, C, n_h, h_p, n_w, w_p]
    outputs = outputs.permute(0, 3, 1, 4, 2, 5).contiguous()

    # -> [B, C, n_h*h_p, n_w*w_p]
    merged = outputs.reshape(B, C, n_h * h_p, n_w * w_p)

    if out_size is None:
        out_size = (H, W)

    return merged


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABN(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABN(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABN(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output

@HEADS.register_module()
class StdcCCHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels,img_size, reduce=False,decoder_flag='cc', **kwargs):
        super(StdcCCHead, self).__init__(**kwargs)
        self.decoder_flag=decoder_flag
        self.patch_size=img_size[0]
        self.down_ratio = down_ratio
        self.stdc_net = ShallowNet_lk(in_channels=3,num_classes=self.num_classes)
        self.reduce = Reducer() if reduce else None
        self.convert_shallow8=nn.Conv2d(256,self.channels,stride=1,kernel_size=1)
        self.convert_shallow16=nn.Conv2d(512,self.channels,stride=1,kernel_size=3,padding=1)
        self.addConv=AddFuse(self.channels,self.channels)
        self.fuse=AddFuse(self.channels,self.channels)
        self.cls_seg8 = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.cca = CrissCrossAttention(self.channels)
        self.rcca=RCCAModule(self.channels,self.channels,self.num_classes)

        # self._freeze_module(self.stdc_net)
        # self._freeze_module(self.reduce)
        # self._freeze_module(self.convert_shallow8)
        # self._freeze_module(self.convert_shallow16)
        # self._freeze_module(self.addConv)
        # self._freeze_module(self.fuse)
        # self._freeze_module(self.cls_seg8)
        # self._freeze_module(self.conv_seg)

    def _freeze_module(self, module):
        if module is not None:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


    def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
        """Forward function."""
        ori_size = inputs.shape[-2:]  # (H, W)
        inputs, patch_meta = split_to_patches(inputs, self.patch_size)
        shallow_feat8, shallow_feat16 = self.stdc_net(inputs)

        # add fusion
        shallow_feat16 = self.convert_shallow16(shallow_feat16)
        shallow_feat8=self.convert_shallow8(shallow_feat8)
        _, _, h, w = shallow_feat8.size()
        shallow_feat16 = F.interpolate(shallow_feat16, size=(h , w ), mode='bilinear', align_corners=False)
        fusion=self.addConv(shallow_feat8,shallow_feat16)
        fusion = merge_patches(fusion, patch_meta, out_size=ori_size)

        if self.decoder_flag=='ori':
            if train_flag==True:
                output = self.cls_seg(fusion)
                aux_output = self.cls_seg8(shallow_feat8)
                aux_output = merge_patches(aux_output, patch_meta, out_size=ori_size)
                return output,aux_output
            else:
                output = self.cls_seg(fusion)
                return output
        if self.decoder_flag=='pid':
            if train_flag==True:
                output = self.cls_seg(fusion)
                predict = self.pid.forward_dual(fusion)
                aux_output = self.cls_seg8(shallow_feat8)
                aux_output = merge_patches(aux_output, patch_meta, out_size=ori_size)
                return output,aux_output,predict
            else:
                predict = self.pid.forward_dual(fusion)
                return predict
        if self.decoder_flag=='cc':
            if train_flag==True:
                # fusion = self.cca(fusion)
                # output = self.cls_seg(fusion)
                output=self.rcca(fusion)
                aux_output = self.cls_seg8(shallow_feat8)
                aux_output = merge_patches(aux_output, patch_meta, out_size=ori_size)

                return output,aux_output
            else:
                output=self.rcca(fusion)
                return output








    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg,mask=None):
        if self.decoder_flag=='ori':
            output,aux_output= self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses= self.losses(output, gt_semantic_seg)
            losses_aux = self.losses(aux_output, gt_semantic_seg)
            return  losses,losses_aux
        elif self.decoder_flag=='pid':
            output,aux_output,predict= self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses = self.losses(predict, gt_semantic_seg)
            losses_middle= self.losses(output, gt_semantic_seg)
            losses_aux = self.losses(aux_output, gt_semantic_seg)
            return  losses,losses_middle,losses_aux
        if self.decoder_flag=='cc':
            output,aux_output= self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses= self.losses(output, gt_semantic_seg)
            losses_aux = self.losses(aux_output, gt_semantic_seg)
            return  losses,losses_aux
        else:
            output,aux_output,f1,final_logits = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses = self.losses(final_logits, gt_semantic_seg)
            losses_s = self.losses(output, gt_semantic_seg)
            losses_aux = self.losses(aux_output, gt_semantic_seg)
            # losses_f1 = self.losses(f1, gt_semantic_seg)
            # losses_f2 = self.losses(f2, gt_semantic_seg)
            # losses_f3 = self.losses(f3, gt_semantic_seg)
            # losses_f4 = self.losses(f4, gt_semantic_seg)
            # losses_f5 = self.losses(f5, gt_semantic_seg)
            return  losses,losses_s,losses_aux



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

