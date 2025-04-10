import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from .diff_fusion import FeatureFusionModule
from .harr import HarrUp
from other_utils.heatmap import save_image,save_heatmap
from mmseg.ops import resize
from mmseg.models.losses.detail_loss import DetailAggregateLoss
from .pid import Bag
from .diff_head import DiffHead
from .diff_point import DiffPoint
from .spnet import SpNet,get_coordsandfeatures,SparseResNet50
from .spnetv2 import SpNetV2
from other_utils.split_tensor import split_tensor,restore_tensor
from .pidnet import PIDNet
from .pidnet_un import PIDNet
from mmseg.models.losses.bce_dice_loss import BCEDiceLoss
from mmseg.models.losses.detail_loss import DetailAggregateLoss



class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        elif len(score) == 1:
            return sb_weights * self._forward(score[0], target)

        else:
            raise ValueError("lengths of prediction and target are not identical!")


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target,name):
        loss=dict()
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        balance_weights = [0.5, 0.5]
        sb_weights = 0.5
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * \
                        (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])

        elif len(score) == 1:
            loss[name+'ohem_loss']=sb_weights * self._ohem_forward(score[0], target)
            return loss

        else:
            raise ValueError("lengths of prediction and target are not identical!")


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
        self.conv2 = ConvModule(channels // 2, channels//2 , 3, stride=1, padding=1, conv_cfg=conv_cfg,
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


class CGM(nn.Module):
    def __init__(self):
        super(CGM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.prob = nn.Sigmoid()

    def forward(self, feature, map):
        map = map[:, 1, :, :].unsqueeze(1)
        m_batchsize, C, height, width = feature.size()
        proj_query = feature.view(m_batchsize, C, -1)
        proj_key = map.view(m_batchsize, 1, -1).permute(0, 2, 1)
        attention = torch.bmm(proj_query, proj_key)
        attention = attention.unsqueeze(2)
        attention = self.prob(attention)
        out = attention * feature
        out = self.gamma * out + feature

        return out


class PSM(nn.Module):
    def __init__(self):
        super(PSM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature, map):
        map = map[:, 1, :, :].unsqueeze(1)
        m_batchsize, C, height, width = feature.size()
        feature_enhance = []
        for i in range(0, C):
            feature_channel = feature[:, i, :, :].unsqueeze(1)
            proj_query = feature_channel.view(m_batchsize, -1, width * height).permute(0, 2, 1)
            proj_key = map.view(m_batchsize, -1, width * height)
            energy = torch.bmm(proj_query, proj_key)
            attention = self.softmax(energy)
            proj_value = feature_channel.view(m_batchsize, -1, width * height)
            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(m_batchsize, 1, height, width)
            feature_enhance.append(out)
        feature_enhance = torch.cat(feature_enhance, dim=1)
        final_feature = self.gamma * feature_enhance + feature
        return final_feature

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DiffDecoder(nn.Module):
    def __init__(self,high_channel,low_channel,out_channel,num_classes):
        super(DiffDecoder, self).__init__()
        self.high_channel = high_channel
        self.low_channel = low_channel
        self.out_channel = out_channel
        self.conv_high = BasicConv2d(high_channel,self.out_channel,3,1,1)
        self.conv_low = BasicConv2d(low_channel,self.out_channel,3,1,1)
        self.conv_fusion = nn.Conv2d(2*self.out_channel,self.out_channel,3,1,1)
        self.seg_out = nn.Conv2d(self.out_channel, num_classes, 1)

    def forward(self,high_feature, low_feature, diff):
        diff = torch.sigmoid(diff)
        high_feature=diff*high_feature+high_feature
        high_feature = F.interpolate(self.conv_high(high_feature), low_feature.size()[2:], mode='bilinear', align_corners=False)

        diff=F.interpolate(diff, low_feature.size()[2:], mode='bilinear', align_corners=False)
        low_feature=diff*low_feature+low_feature
        low_feature=self.conv_low(low_feature)

        feature_cat=torch.concat((high_feature,low_feature),dim=1)
        fusion=self.conv_fusion(feature_cat)
        predict=self.seg_out(fusion)
        return fusion,predict

class Decoder(nn.Module):
    def __init__(self,high_channel,middle_channel,out_channel,num_classes):
        super(Decoder, self).__init__()
        self.high_channel = high_channel
        self.middle_channel = middle_channel
        self.out_channel = out_channel
        self.conv1 = BasicConv2d(high_channel,self.middle_channel,3,1,1)
        self.conv2 = BasicConv2d(self.middle_channel,self.middle_channel,3,1,1)
        self.conv3= BasicConv2d(self.middle_channel,self.out_channel,3,1,1)
        # self.seg_out = nn.Conv2d(self.out_channel, num_classes, 1)

    def forward(self,high_feature,diff):
        diff = torch.sigmoid(diff)
        high_feature = diff * high_feature + high_feature
        high_feature=self.conv1(high_feature)
        high_feature=self.conv2(high_feature)
        high_feature=self.conv3(high_feature)
        # predict=self.seg_out(high_feature)
        # high_feature = F.interpolate(high_feature, scale_factor=2, mode='bilinear', align_corners=False)
        # predict = F.interpolate(predict, scale_factor=2, mode='bilinear', align_corners=False)

        return high_feature




@HEADS.register_module()
class PidUnHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels,img_size, reduce=False,decoder_flag=2, **kwargs):
        super(PidUnHead, self).__init__(**kwargs)
        self.decoder_flag=decoder_flag
        self.down_ratio = down_ratio
        self.pid=PIDNet(m=2, n=3, num_classes=self.num_classes, planes=32, ppm_planes=96, head_planes=128, augment=True)
        self.detail_loss=DetailAggregateLoss()
        # self.psm=PSM()
        # self.cgm=CGM()
        # self.diff_decoder1=DiffDecoder(128,64,64,self.num_classes)
        # self.diff_decoder2=DiffDecoder(64,64,64,self.num_classes)
        # self.diff_decoder3=DiffDecoder(64,64,32,self.num_classes)
        # self.diff_decoder4=DiffDecoder(32,32,32,self.num_classes)
        # self.diff_decoder5=DiffDecoder(32,32,32,self.num_classes)
        #
        self.decoder1=Decoder(128,128,128,self.num_classes)
        self.decoder2=Decoder(128,128,64,self.num_classes)
        self.decoder3=Decoder(64,64,32,self.num_classes)
        #
        self.seg_out=nn.Conv2d(in_channels=128,out_channels=self.num_classes,kernel_size=1,stride=1)
        #
        #
        self.diff1 = DiffHead(in_channels=1, in_index=3, channels=32, dropout_ratio=0.1,
                                     num_classes=self.num_classes, align_corners=False, loss_decode=dict(
                type='BCEDiceLoss'))
        self.diff2 = DiffHead(in_channels=1, in_index=3, channels=32, dropout_ratio=0.1,
                                     num_classes=self.num_classes, align_corners=False, loss_decode=dict(
                type='BCEDiceLoss'))
        self.diff3 = DiffHead(in_channels=1, in_index=3, channels=32, dropout_ratio=0.1,
                                     num_classes=self.num_classes, align_corners=False, loss_decode=dict(
                type='BCEDiceLoss'))
        # self.diff4 = DiffHead(in_channels=1, in_index=3, channels=32, dropout_ratio=0.1,
        #                              num_classes=self.num_classes, align_corners=False, loss_decode=dict(
        #         type='BCEDiceLoss'))
        # self.diff5 = DiffHead(in_channels=1, in_index=3, channels=32, dropout_ratio=0.1,
        #                              num_classes=self.num_classes, align_corners=False, loss_decode=dict(
        #         type='BCEDiceLoss'))
        self.ohem=OhemCrossEntropy()
        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
        self.conv_seg_p = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, 64,
                                         32, self.num_classes, kernel_size=1)
        self.reduce = Reducer() if reduce else None
        self.diff_flag=False
        # self.bag=Bag(512,128)


    def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
        """Forward function."""
        decoder_flag=self.decoder_flag
        prymaid_results = self.lap_prymaid_conv.pyramid_decom(inputs)
        high_residual_1 = prymaid_results[0]
        high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
                                        align_corners=False)
        high_residual_input = torch.cat([high_residual_1, high_residual_2], dim=1)
        x_extra_p, predict, fusion = self.pid.forward_t(high_residual_input)

        if decoder_flag==1:
            predict_down = F.interpolate(predict, fusion.size()[2:], mode='bilinear', align_corners=False)
            if train_flag:
                loss_diff1, _, diff_pred1 = self.diff1.forward_train_diff(predict_down, img_metas,gt, train_cfg)
                fusion1,pred1=self.diff_decoder1(fusion,feats[-1],diff_pred1)
                loss_diff2, _, diff_pred2 = self.diff2.forward_train_diff(pred1, img_metas, gt, train_cfg)
                fusion2,pred2=self.diff_decoder2(fusion1,feats[-2],diff_pred2)
                loss_diff3, _, diff_pred3 = self.diff3.forward_train_diff(pred2, img_metas, gt, train_cfg)
                fusion3,pred3=self.diff_decoder3(fusion2,feats[-3],diff_pred3)
                loss_diff4, _, diff_pred4 = self.diff4.forward_train_diff(pred3, img_metas, gt, train_cfg)
                fusion4,pred4=self.diff_decoder4(fusion3,feats[-4],diff_pred4)
                loss_diff5, _, diff_pred5 = self.diff5.forward_train_diff(pred4, img_metas, gt, train_cfg)
                fusion5,pred5=self.diff_decoder4(fusion4,feats[-5],diff_pred5)


                return x_extra_p, predict,pred5,pred4,pred3,pred2,pred1 ,loss_diff5,loss_diff4,loss_diff3,loss_diff2,loss_diff1
            else:
                diff_pred1 = self.diff1.forward_test_diff(predict_down)
                fusion1, pred1 = self.diff_decoder1(fusion, feats[-1], diff_pred1)
                diff_pred2 = self.diff2.forward_test_diff(pred1)
                fusion2, pred2 = self.diff_decoder2(fusion1, feats[-2], diff_pred2)
                diff_pred3 = self.diff3.forward_test_diff(pred2)
                fusion3, pred3 = self.diff_decoder3(fusion2, feats[-3], diff_pred3)
                diff_pred4 = self.diff4.forward_test_diff(pred3)
                fusion4, pred4 = self.diff_decoder4(fusion3, feats[-4], diff_pred4)
                diff_pred5 = self.diff5.forward_test_diff(pred4)
                fusion5,pred5=self.diff_decoder4(fusion4,feats[-5],diff_pred5)


                return pred5
        elif decoder_flag==2:

            if train_flag:
                loss_diff1, _, diff_pred1 = self.diff1.forward_train_diff(predict, img_metas, gt, train_cfg)
                fusion1 = self.decoder1(fusion, diff_pred1)
                output=self.seg_out(fusion1)
                # loss_diff2, _, diff_pred2 = self.diff2.forward_train_diff(predict1, img_metas, gt, train_cfg)
                # fusion2,predict2 = self.decoder2(fusion1, diff_pred2)
                # loss_diff3, _, diff_pred3 = self.diff3.forward_train_diff(predict2, img_metas, gt, train_cfg)
                # fusion3,predict3 = self.decoder3(fusion2, diff_pred3)
                return x_extra_p,predict,output,loss_diff1
            else:
                diff_pred1 = self.diff1.forward_test_diff(predict)
                fusion1 = self.decoder1(fusion, diff_pred1)
                output = self.seg_out(fusion1)
                # diff_pred2 = self.diff2.forward_test_diff(predict1)
                # fusion2,predict2 = self.decoder2(fusion1, diff_pred2)
                # diff_pred3 = self.diff3.forward_test_diff(predict2)
                # fusion3,predict3 = self.decoder3(fusion2, diff_pred3)
                return output
        else:
            if train_flag:
                return x_extra_p, predict
            else:
                return predict









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

    def weighted_bce(self,bd_pre, target):
        n, c, h, w = bd_pre.size()
        log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
        target_t = target.view(1, -1)

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)

        weight = torch.zeros_like(log_p)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

        return loss

    def boundary_loss(self,pred,gt):
        edges = []
        B, C, H, W = gt.shape
        for b in range(B):
            single_image = gt[b, 0]
            edge = cv2.Canny(single_image.cpu().numpy().astype(np.uint8), 0.1, 0.2)
            kernel = np.ones((4, 4), np.uint8)
            edge = (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0
            edges.append(torch.from_numpy(edge).float().cuda())
        edges = torch.stack(edges, dim=0)

        pred=F.interpolate(pred, gt.size()[2:], mode='bilinear', align_corners=False)

        pred_cls=torch.sigmoid(pred)
        pred_cls=(pred_cls>0.5).float()
        save_image(pred_cls[0].squeeze().detach().cpu().numpy(), filename='boundary_pred',
                   save_dir='/root/autodl-tmp/isdnet_harr/diff_dir', )
        save_image(edges[0].detach().cpu().numpy(), filename='boundary_gt',
                   save_dir='/root/autodl-tmp/isdnet_harr/diff_dir', )

        loss = dict()
        # loss['boundary_loss']=20*self.weighted_bce(pred,edges)
        loss['boundary_loss']=0.5*self.bce_dice(pred,edges)
        return loss

    # def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg,mask=None):
    #
    #     output,x,temp_p,loss_shallow_diff,x4_cls= self.forward(inputs, prev_output,
    #                                                                                          mask=mask,
    #                                                                                          gt=gt_semantic_seg,
    #                                                                                          img_metas=img_metas,
    #                                                                                          train_cfg=train_cfg,
    #                                                                                          diff_pred_deep=mask)
    #     losses = self.losses(output, gt_semantic_seg)
    #     losses_deep = self.losses(temp_p, gt_semantic_seg)
    #     losses_x4 = self.losses(x4_cls, gt_semantic_seg)
    #     # losses_aux16 = self.losses(seg_logits_aux16, gt_semantic_seg)
    #     # losses_aux16 = torch.tensor(0)
    #
    #     # last_losses = self.losses(last_output, gt_semantic_seg)
    #
    #     # losses_aux8 = self.losses(seg_logits_aux8, gt_semantic_seg)
    #     # losses_detail = self.detail_loss(shallow_feat4_cls,gt_semantic_seg.squeeze(1) )
    #     return losses,losses_deep, loss_shallow_diff,losses_x4


    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg,mask=None):
        if self.decoder_flag==1:
            x_extra_p, output,pred5,pred4,pred3,pred2,pred1 ,loss_diff5,loss_diff4,loss_diff3,loss_diff2,loss_diff1= self.forward(inputs, prev_output,
                                                                                                 mask=mask,
                                                                                                 gt=gt_semantic_seg,
                                                                                                 img_metas=img_metas,
                                                                                                 train_cfg=train_cfg,
                                                                                                 diff_pred_deep=mask)

            output=F.interpolate(output, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            x_extra_p=F.interpolate(x_extra_p, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            pred5 = F.interpolate(pred5, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            pred4=F.interpolate(pred4, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            pred3=F.interpolate(pred3, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            pred2=F.interpolate(pred2, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            pred1=F.interpolate(pred1, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            losses=self.ohem(output,gt_semantic_seg.squeeze(1),'middle')
            losses_p = self.ohem(x_extra_p, gt_semantic_seg.squeeze(1),'p_')
            losses_pred5 = self.ohem(pred5, gt_semantic_seg.squeeze(1),'pred5_')
            losses_pred4 = self.ohem(pred4, gt_semantic_seg.squeeze(1),'pred4_')
            losses_pred3 = self.ohem(pred3, gt_semantic_seg.squeeze(1),'pred3_')
            losses_pred2 = self.ohem(pred2, gt_semantic_seg.squeeze(1),'pred2_')
            losses_pred1 = self.ohem(pred1, gt_semantic_seg.squeeze(1),'pred1_')

            # losses_p = self.losses(x_extra_p, gt_semantic_seg)
            # losses = self.losses(output, gt_semantic_seg)
            # losses_d=self.boundary_loss(x_extra_d,gt_semantic_seg)

            return losses,losses_p\
                ,losses_pred5,losses_pred4,losses_pred3,losses_pred2,losses_pred1\
                ,loss_diff5,loss_diff4,loss_diff3,loss_diff2,loss_diff1
        elif self.decoder_flag==2:
            x_extra_p,predict,output,loss_diff1 = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)

            output = F.interpolate(output, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            predict = F.interpolate(predict, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            # predict1 = F.interpolate(predict1, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            # predict2 = F.interpolate(predict2, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            x_extra_p = F.interpolate(x_extra_p, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            losses = self.ohem(output, gt_semantic_seg.squeeze(1), 'output_')
            losses_p = self.ohem(x_extra_p, gt_semantic_seg.squeeze(1), 'p_')
            losses_middle = self.ohem(predict, gt_semantic_seg.squeeze(1), 'middle_')
            # losses_predict1 = self.ohem(predict1, gt_semantic_seg.squeeze(1), 'predict1_')
            # losses_predict2 = self.ohem(predict2, gt_semantic_seg.squeeze(1), 'predict2_')


            return losses,losses_middle,losses_p,loss_diff1


        else:
            x_extra_p, predict  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)

            predict = F.interpolate(predict, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            x_extra_p = F.interpolate(x_extra_p, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            # losses_boundary = self.detail_loss(x8_cls, gt_semantic_seg.squeeze(1))
            losses_p = self.ohem(x_extra_p, gt_semantic_seg.squeeze(1), 'p_')
            losses = self.ohem(predict, gt_semantic_seg.squeeze(1), 'middle_')
            return losses, losses_p

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
