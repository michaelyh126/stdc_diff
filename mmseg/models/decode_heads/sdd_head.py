import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from .sdd_stdc_head import ShallowNet
from .diff_fusion import FeatureFusionModule
from .harr import HarrUp
from mmseg.ops import resize
from mmseg.models.losses.detail_loss import DetailAggregateLoss
from .pid import Bag
from .diff_head import DiffHead
from .diff_point import DiffPoint

class SegmentationHead(nn.Module):
    def __init__(self, conv_cfg, norm_cfg, act_cfg, in_channels, mid_channels, n_classes, *args, **kwargs):
        super(SegmentationHead, self).__init__()

        self.conv_bn_relu = ConvModule(in_channels, mid_channels, 3,
                                       stride=1,
                                       padding=1,
                                       conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)

        self.conv_out = nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.conv_out(x)
        return x


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, gauss_chl=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.gauss_chl = gauss_chl
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda')):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(self.gauss_chl, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

    # def pyramid_img_decom(self, img):
    #     pyr = []
    #     current = img
    #     for _ in range(self.num_high):
    #         current = self.downsample(current)
    #         pyr.append(current)
    #     return pyr
    #
    # def pyramid_recons(self, pyr):
    #     image = pyr[-1]
    #     for level in reversed(pyr[:-1]):
    #         up = self.upsample(image)
    #         if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
    #             up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
    #         image = up + level
    #     return image


class SRDecoder(nn.Module):
    # super resolution decoder
    def __init__(self, conv_cfg, norm_cfg, act_cfg, channels=128, up_lists=[2, 2, 2]):
        super(SRDecoder, self).__init__()
        self.conv1 = ConvModule(channels, channels // 2, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up1 = nn.Upsample(scale_factor=up_lists[0])
        self.conv2 = ConvModule(channels // 2, channels // 2, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up2 = nn.Upsample(scale_factor=up_lists[1])
        self.conv3 = ConvModule(channels // 2, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up3 = nn.Upsample(scale_factor=up_lists[2])
        self.conv_sr = SegmentationHead(conv_cfg, norm_cfg, act_cfg, channels, channels // 2, 3, kernel_size=1)

    def forward(self, x, fa=False):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        feats = self.conv3(x)
        outs = self.conv_sr(feats)
        if fa:
            return feats, outs
        else:
            return outs


class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super(ChannelAtt, self).__init__()
        self.conv_bn_relu = ConvModule(in_channels, out_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_1x1 = ConvModule(out_channels, out_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x, fre=False):
        """Forward function."""
        feat = self.conv_bn_relu(x)
        if fre:
            h, w = feat.size()[2:]
            h_tv = torch.pow(feat[..., 1:, :] - feat[..., :h - 1, :], 2)
            w_tv = torch.pow(feat[..., 1:] - feat[..., :w - 1], 2)
            atten = torch.mean(h_tv, dim=(2, 3), keepdim=True) + torch.mean(w_tv, dim=(2, 3), keepdim=True)
        else:
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_1x1(atten)
        return feat, atten


class RelationAwareFusion(nn.Module):
    def __init__(self, channels, conv_cfg, norm_cfg, act_cfg, ext=2, r=16):
        super(RelationAwareFusion, self).__init__()
        self.r = r
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))
        self.spatial_mlp = nn.Sequential(nn.Linear(channels * 2, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.spatial_att = ChannelAtt(channels * ext, channels, conv_cfg, norm_cfg, act_cfg)
        self.context_mlp = nn.Sequential(*[nn.Linear(channels * 2, channels), nn.ReLU(), nn.Linear(channels, channels)])
        self.context_att = ChannelAtt(channels, channels, conv_cfg, norm_cfg, act_cfg)
        self.context_head = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)
        self.smooth = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, sp_feat, co_feat):
        # **_att: B x C x 1 x 1
        s_feat, s_att = self.spatial_att(sp_feat)
        c_feat, c_att = self.context_att(co_feat)
        b, c, h, w = s_att.size()
        s_att_split = s_att.view(b, self.r, c // self.r)
        c_att_split = c_att.view(b, self.r, c // self.r)
        chl_affinity = torch.bmm(s_att_split, c_att_split.permute(0, 2, 1))
        chl_affinity = chl_affinity.view(b, -1)
        sp_mlp_out = F.relu(self.spatial_mlp(chl_affinity))
        co_mlp_out = F.relu(self.context_mlp(chl_affinity))
        re_s_att = torch.sigmoid(s_att + self.g1 * sp_mlp_out.unsqueeze(-1).unsqueeze(-1))
        re_c_att = torch.sigmoid(c_att + self.g2 * co_mlp_out.unsqueeze(-1).unsqueeze(-1))
        c_feat = torch.mul(c_feat, re_c_att)
        s_feat = torch.mul(s_feat, re_s_att)
        c_feat = F.interpolate(c_feat, s_feat.size()[2:], mode='bilinear', align_corners=False)
        c_feat = self.context_head(c_feat)
        # s_feat=F.interpolate(s_feat,c_feat.size()[2:],mode='bilinear', align_corners=False)
        # s_feat=self.context_head(s_feat)
        out = self.smooth(s_feat + c_feat)
        return s_feat, c_feat, out


class Reducer(nn.Module):
    # Reduce channel (typically to 128)
    def __init__(self, in_channels=512, reduce=128, bn_relu=True):
        super(Reducer, self).__init__()
        self.bn_relu = bn_relu
        self.conv1 = nn.Conv2d(in_channels, reduce, 1, bias=False)
        if self.bn_relu:
            self.bn1 = nn.BatchNorm2d(reduce)

    def forward(self, x):

        x = self.conv1(x)
        if self.bn_relu:
            x = self.bn1(x)
            x = F.relu(x)

        return x


@HEADS.register_module()
class SddHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels, reduce=False, **kwargs):
        super(SddHead, self).__init__(**kwargs)
        self.down_ratio = down_ratio
        self.fuse8 = RelationAwareFusion(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg, ext=2)
        self.fuse16 = RelationAwareFusion(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg, ext=4)
        self.sr_decoder = SRDecoder(self.conv_cfg, self.norm_cfg, self.act_cfg,
                                    channels=self.channels, up_lists=[4, 2, 2])
        # shallow branch
        self.stdc_net = ShallowNet(in_channels=6, pretrain_model="/root/autodl-tmp/pretrained models/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        self.diff_fusion1=FeatureFusionModule(256,128,sp_channel=512,cp_channel=128)
        self.diff_fusion2=FeatureFusionModule(256,128,sp_channel=128,cp_channel=128)
        # self.harr_up=HarrUp()
        self.shallow_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
                type='BCEDiceLoss'))
        self.fuse_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
                type='BCEDiceLoss'))
        # self.detail_loss=DetailAggregateLoss()
        # self.diff_point_shallow=DiffPoint(in_channels=[256],channels=256,num_classes=19,align_corners=False,in_index=[0])
        # self.diff_point_fuse=DiffPoint(in_channels=[256],channels=256,num_classes=19,align_corners=False,in_index=[0])
        # self.bag=Bag(128,128)
        self.refine_mlp=ConvModule(512,self.num_classes,1,1)


        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
        self.conv_seg_aux_16 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                                self.channels // 2, self.num_classes, kernel_size=1)
        self.conv_seg_aux_8 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                               self.channels // 2, self.num_classes, kernel_size=1)
        self.conv_seg = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                         self.channels // 2, self.num_classes, kernel_size=1)

        self.reduce = Reducer() if reduce else None


    def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
        """Forward function."""
        prymaid_results = self.lap_prymaid_conv.pyramid_decom(inputs)
        high_residual_1 = prymaid_results[0]
        high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
                                        align_corners=False)
        high_residual_input = torch.cat([high_residual_1, high_residual_2], dim=1)
        # mask_up = resize(
        #     input=mask,
        #     size=inputs.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # mask_up=torch.sigmoid(mask_up)
        # mask_up = (mask_up > 0.5).float()
        # high_residual_input=torch.mul(high_residual_input,mask_up)


        shallow_feat8, shallow_feat16,feat16_cls = self.stdc_net(high_residual_input)
        deep_feat = prev_output[0]
        # deep_feat_up=self.harr_up(deep_feat)
        if self.reduce is not None:
            deep_feat = self.reduce(deep_feat)


        if train_flag:
            loss_shallow_diff, diff_map, diff_pred_shallow = self.shallow_diff.forward_train_diff(feat16_cls, img_metas, gt, train_cfg)

            _, aux_feat16, fused_feat_16 = self.diff_fusion1(shallow_feat16, deep_feat,diff_pred_deep)
            _, aux_feat8, fused_feat_8 = self.fuse8(shallow_feat8, fused_feat_16)
            _, aux_feat8, fused_feat_8 = self.diff_fusion2(fused_feat_8, fused_feat_16,diff_pred_shallow)

            output = self.cls_seg(fused_feat_8)


            output_aux16 = self.conv_seg_aux_16(aux_feat8)
            output_aux8 = self.conv_seg_aux_8(aux_feat16)
            feats, output_sr = self.sr_decoder(deep_feat, True)
            losses_re = self.image_recon_loss(high_residual_1 + high_residual_2, output_sr, re_weight=0.1)
            losses_fa = self.feature_affinity_loss(deep_feat, feats)
            # losses_detail = self.detail_loss(shallow_feat4_cls,)
            return output, output_aux16, output_aux8, losses_re, losses_fa,loss_shallow_diff
        else:
            diff_pred_shallow=self.shallow_diff.forward_test_diff(feat16_cls, img_metas)
            # _, aux_feat16, fused_feat_16 = self.fuse16(shallow_feat16, deep_feat)
            # _, aux_feat8, fused_feat_8 = self.fuse8(shallow_feat8, fused_feat_16)
            _, aux_feat16, fused_feat_16 = self.diff_fusion1(shallow_feat16, deep_feat,diff_pred_deep)
            _, aux_feat8, fused_feat_8 = self.fuse8(shallow_feat8, fused_feat_16)
            _, aux_feat8, fused_feat_8 = self.diff_fusion2(fused_feat_8, fused_feat_16,diff_pred_shallow)
            # refine_fused_feat_8 = self.bag(shallow_feat16, fused_feat_8, mask)
            output = self.cls_seg(fused_feat_8)
            # output = self.dysampler(output)
            return output

    def image_recon_loss(self, img, pred, re_weight=0.5):
        loss = dict()
        if pred.size()[2:] != img.size()[2:]:
            pred = F.interpolate(pred, img.size()[2:], mode='bilinear', align_corners=False)
        recon_loss = F.mse_loss(pred, img) * re_weight
        loss['recon_losses'] = recon_loss
        return loss

    def feature_affinity_loss(self, seg_feats, sr_feats, fa_weight=1., eps=1e-6):
        if seg_feats.size()[2:] != sr_feats.size()[2:]:
            sr_feats = F.interpolate(sr_feats, seg_feats.size()[2:], mode='bilinear', align_corners=False)
        loss = dict()
        # flatten:
        seg_feats_flatten = torch.flatten(seg_feats, start_dim=2)
        sr_feats_flatten = torch.flatten(sr_feats, start_dim=2)
        # L2 norm
        seg_norm = torch.norm(seg_feats_flatten, p=2, dim=2, keepdim=True)
        sr_norm = torch.norm(sr_feats_flatten, p=2, dim=2, keepdim=True)
        # similiarity
        seg_feats_flatten = seg_feats_flatten / (seg_norm + eps)
        sr_feats_flatten = sr_feats_flatten / (sr_norm + eps)
        seg_sim = torch.matmul(seg_feats_flatten.permute(0, 2, 1), seg_feats_flatten)
        sr_sim = torch.matmul(sr_feats_flatten.permute(0, 2, 1), sr_feats_flatten)
        # L1 loss
        loss['fa_loss'] = F.l1_loss(seg_sim, sr_sim.detach()) * fa_weight
        return loss

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg,mask=None):
        seg_logits, seg_logits_aux16, seg_logits_aux8, losses_recon, losses_fa ,loss_shallow_diff= self.forward(inputs, prev_output,mask=mask,gt=gt_semantic_seg,img_metas=img_metas,train_cfg=train_cfg,diff_pred_deep=mask)
        losses = self.losses(seg_logits, gt_semantic_seg)
        losses_aux16 = self.losses(seg_logits_aux16, gt_semantic_seg)
        losses_aux8 = self.losses(seg_logits_aux8, gt_semantic_seg)
        # losses_detail = self.detail_loss(shallow_feat4_cls,gt_semantic_seg.squeeze(1) )
        return losses, losses_aux16, losses_aux8, losses_recon, losses_fa,loss_shallow_diff

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
