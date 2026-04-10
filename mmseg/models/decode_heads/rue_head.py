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
from .pidnet_single import PIDNet
# from .pidnet import PIDNet
# from .pidnet_un import PIDNet
# from .pidnet_distill import PIDNet
# from .pidnet_stdc import PIDNet
from mmseg.models.losses.bce_dice_loss import BCEDiceLoss
from mmseg.models.losses.detail_loss import DetailAggregateLoss
from .isdhead import RelationAwareFusion
from mmseg.models.sampler.dysample import DySample
from other_utils.histogram import tensor_histogram
from mmseg.models.decode_heads.isdhead import SRDecoder
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def merge_patches_to_image(patches_list, rows=8, cols=8):
    """
    将 patch 列表还原为完整图像
    Args:
        patches_list: list of tensors, 每个tensor形状为 [B, C, h, w]
        rows: 网格行数
        cols: 网格列数
    Returns:
        image: tensor, 形状为 [B, C, H, W]
    """
    # 确保提供的块数量足够
    assert len(patches_list) >= rows * cols, "Patches数量不足"

    # 获取每个块的大小
    # patches_list[0] 形状: [B, C, h, w]
    b, c, h, w = patches_list[0].shape

    # 初始化存放每一行结果的列表
    image_rows = []

    # 按行遍历
    for i in range(rows):
        # 取出当前行的所有块
        # split函数是按行优先顺序存储的，所以第 i 行的索引范围是 [i*cols : (i+1)*cols]
        row_patches = patches_list[i * cols: (i + 1) * cols]

        # 将当前行的块在宽度维度上进行拼接 (dim=-1)
        # 拼接后形状: [B, C, h, W]
        row_tensor = torch.cat(row_patches, dim=-1)
        image_rows.append(row_tensor)

    # 将所有行在高度维度上进行拼接 (dim=-2)
    # 拼接后形状: [B, C, H, W]
    image = torch.cat(image_rows, dim=-2)

    return image


def split_to_grid_patches(img, rows=8, cols=8):
    _, _, H, W = img.shape
    h_step = H // rows
    w_step = W // cols
    patches_container = []
    for i in range(rows):
        for j in range(cols):
            h_start = i * h_step
            h_end = (i + 1) * h_step
            w_start = j * w_step
            w_end = (j + 1) * w_step
            patch = img[:, :, h_start:h_end, w_start:w_end]
            patches_container.append(patch)

    return patches_container

@HEADS.register_module()
class RUEHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels,img_size, reduce=False,decoder_flag='aff', **kwargs):
        super(RUEHead, self).__init__(**kwargs)
        self.decoder_flag=decoder_flag
        self.shallow_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
                type='BCEDiceLoss'))
        self.down_ratio = down_ratio
        self.convert_shallow8=nn.Conv2d(256,self.channels,stride=1,kernel_size=1)
        self.cls_fd=nn.Conv2d(128,self.channels,stride=1,kernel_size=1)
        self.stdc_net = ShallowNet(in_channels=3, pretrain_model="/root/autodl-tmp/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        self.reduce = Reducer() if reduce else None
        self.convert_shallow16=nn.Conv2d(512,self.channels,stride=1,kernel_size=3,padding=1)
        self.addConv=AddFuse(self.channels,self.channels)
        self.fuse=AddFuse(self.channels,self.channels)

    def forward(self, fd, img, patch_index, estimation, train_flag=True, mask=None, gt=None,
                img_metas=None, train_cfg=None, diff_pred_deep=None):
        """Forward function."""

        # ✅ 修复3：推理时关闭梯度
        if not train_flag:
            torch.set_grad_enabled(False)

        try:
            _, _, h, w = img.size()
            n_h = math.ceil(h / 64) * 8
            n_w = math.ceil(w / 64) * 8
            indices = patch_index[0, 0]

            img_patches = split_to_grid_patches(img)
            fd_up = F.interpolate(fd, size=(n_h, n_w), mode='bilinear', align_corners=False)
            estimation = F.interpolate(estimation, size=(n_h, n_w), mode='bilinear', align_corners=False)
            fd_patches = split_to_grid_patches(fd_up)
            estimation_patches = split_to_grid_patches(estimation)

            # ✅ 修复1：将所有需要处理的 patch 收集成 batch，一次过网络
            selected_indices = [idx.item() for idx in indices]
            batch_patches = torch.cat([img_patches[i].unsqueeze(0) for i in selected_indices], dim=0)
            batch_fd = torch.cat([fd_patches[i].unsqueeze(0) for i in selected_indices], dim=0)
            batch_est = torch.cat([estimation_patches[i].unsqueeze(0) for i in selected_indices], dim=0)
            N,B,C,H,W = batch_patches.size()  # patch 数量
            batch_patches = batch_patches.view(N * B, C, H, W)
            batch_est = batch_est.view(N * B, -1, batch_est.size(3), batch_est.size(4))
            batch_fd = batch_fd.view(N * B, -1, batch_fd.size(3), batch_fd.size(4))
            # 一次前向传播，GPU 并行处理所有 patch
            shallow_feat8, shallow_feat16 = self.stdc_net(batch_patches)
            shallow_feat16 = self.convert_shallow16(shallow_feat16)
            shallow_feat8 = self.convert_shallow8(shallow_feat8)

            _, _, feat_h, feat_w = shallow_feat8.size()
            shallow_feat16 = F.interpolate(shallow_feat16, size=(feat_h, feat_w),
                                           mode='bilinear', align_corners=False)

            fusion = self.addConv(shallow_feat8, shallow_feat16)

            # ✅ 修复2：立即写回并释放中间变量
            updated = fusion * batch_est + (1 - batch_est) * batch_fd
            updated = updated.view(N, -1, updated.size(1), updated.size(2), updated.size(3))
            for j, i in enumerate(selected_indices):
                fd_patches[i] = updated[j]

            # 显式释放不再需要的中间变量
            del batch_patches, batch_fd, batch_est
            del shallow_feat8, shallow_feat16, fusion, updated

            out = merge_patches_to_image(fd_patches)
            output = self.cls_seg(out)
            fd_seg = self.cls_fd(fd)

            if train_flag:
                return output, fd_seg
            else:
                return output
        finally:
            if not train_flag:
                torch.set_grad_enabled(True)

    # def forward(self, fd, img, patch_index,estimation, train_flag=True, mask=None, gt=None, img_metas=None,
    #             train_cfg=None, diff_pred_deep=None):
    #     """Forward function."""
    #     _,_,h,w=img.size()
    #     n_h=math.ceil(h/64)*8
    #     n_w=math.ceil(w/64)*8
    #     indices = patch_index[0, 0]
    #     img_patches=split_to_grid_patches(img)
    #     fd_up=F.interpolate(fd, size=(n_h, n_w), mode='bilinear', align_corners=False)
    #     estimation=F.interpolate(estimation, size=(n_h, n_w), mode='bilinear', align_corners=False)
    #     fd_patches = split_to_grid_patches(fd_up)
    #     estimation_patches = split_to_grid_patches(estimation)
    #     outputs_list = []
    #
    #     for idx in indices:
    #         i = idx.item()
    #         patch = img_patches[i]
    #         fd_patch=fd_patches[i]
    #         estimation_patch=estimation_patches[i]
    #
    #         # --- 核心处理逻辑 (inputs 替换为 patch) ---
    #         # 注意：确保 self.stdc_net 等模块能处理 patch 的尺寸
    #
    #         # 如果需要用到 prev_output，请确保它已定义，这里保留原逻辑
    #         # fd = F.interpolate(prev_output[0], size=(h, w), ...)
    #
    #         shallow_feat8, shallow_feat16 = self.stdc_net(patch)
    #
    #         shallow_feat16 = self.convert_shallow16(shallow_feat16)
    #         shallow_feat8 = self.convert_shallow8(shallow_feat8)
    #
    #         _, _, h, w = shallow_feat8.size()
    #         shallow_feat16 = F.interpolate(shallow_feat16, size=(h, w), mode='bilinear', align_corners=False)
    #
    #         fusion = self.addConv(shallow_feat8, shallow_feat16)
    #         fd_patches[i]=fusion*estimation_patch+(1-estimation_patch)*fd_patch
    #
    #     out=merge_patches_to_image(fd_patches)
    #     output=self.cls_seg(out)
    #     fd_seg=self.cls_fd(fd)
    #     if train_flag==True:
    #         return output,fd_seg
    #     else:
    #         return output

    def forward_train(self, fd, img, patch_index,estimation, img_metas, gt_semantic_seg, train_cfg,mask=None):
        output,fd_seg = self.forward(
            fd, img, patch_index,estimation,
            mask=mask,
            gt=gt_semantic_seg,
            img_metas=img_metas,
            train_cfg=train_cfg,
            diff_pred_deep=mask)
        losses = self.losses(output, gt_semantic_seg)
        losses_aux= self.losses(fd_seg, gt_semantic_seg)
        return  losses,losses_aux



    def forward_test(self, fd, img, patch_index,estimation, img_metas, test_cfg,mask=None):
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

        return self.forward(fd, img, patch_index,estimation, False,diff_pred_deep=mask)

