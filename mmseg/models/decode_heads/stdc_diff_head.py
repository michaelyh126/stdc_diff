import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from .sdd_stdc_head import ShallowNet,ShallowNet_diff
from .diff_fusion import FeatureFusionModule
from .harr import HarrUp
from mmseg.ops import resize
from mmseg.models.losses.detail_loss import DetailAggregateLoss
from .pid import Bag
from .diff_head import DiffHead
from .diff_point import DiffPoint
from .spnet import SpNet,get_coordsandfeatures,SparseResNet50
from .spnetv2 import SpNetV2
from other_utils.split_tensor import split_tensor,restore_tensor
from .FreqFusion import FreqFusion

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
class STDCDiffHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels,img_size, reduce=False, **kwargs):
        super(STDCDiffHead, self).__init__(**kwargs)
        self.down_ratio = down_ratio

        # self.stdc_net = ShallowNet(in_channels=6, pretrain_model="/root/autodl-tmp/pretrained models/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        self.stdc_net = ShallowNet(in_channels=3, pretrain_model="/root/autodl-tmp/pretrained models/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        # self.stdc_net_diff = ShallowNet_diff(in_channels=6, pretrain_model="/root/autodl-tmp/pretrained models/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        self.shallow_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
                type='BCEDiceLoss'))
        self.freqfusion=FreqFusion(128,128)
        self.addConv=nn.Conv2d(128,self.channels,stride=1,kernel_size=3,padding=1)

        # self.fuse_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
        #         type='BCEDiceLoss'))
        # self.detail_loss=DetailAggregateLoss()


        self.convert_shallow16=nn.Conv2d(512,self.channels,stride=1,kernel_size=1)
        self.convert_shallow8=nn.Conv2d(256,self.channels,stride=1,kernel_size=1)
        self.fuse8 = RelationAwareFusion(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg, ext=2)
        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
        self.conv_seg_aux_16 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                                self.channels // 2, self.num_classes, kernel_size=1)
        self.conv_seg_aux_8 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                               self.channels // 2, self.num_classes, kernel_size=1)
        self.conv_seg = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                         self.channels // 2, self.num_classes, kernel_size=1)

        self.reduce = Reducer() if reduce else None
        # self.spnet = SpNet(img_size, num_classes=self.num_classes)
        # self.spnet = SparseResNet50(img_size, num_classes=self.num_classes)
        self.diff_flag=3
        self.sp_fuse=ConvModule(in_channels=self.num_classes*2,out_channels=self.num_classes,kernel_size=3,stride=1,padding=1)
        # self.bag=Bag(self.num_classes,self.num_classes)


    def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
        """Forward function."""

        # prymaid_results = self.lap_prymaid_conv.pyramid_decom(inputs)
        # high_residual_1 = prymaid_results[0]
        # high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
        #                                 align_corners=False)
        # high_residual_input = torch.cat([high_residual_1, high_residual_2], dim=1)
        #
        # shallow_feat8, shallow_feat16 = self.stdc_net(high_residual_input)
        # shallow_feat16=self.convert_shallow16(shallow_feat16)
        # # _, aux_feat8, fused_feat_8 = self.fuse8(shallow_feat8, shallow_feat16)
        # output = self.cls_seg(shallow_feat16)
        # # output_aux16 = self.conv_seg_aux_16(aux_feat8)


        # # 无高频残差无raf
        # shallow_feat8, shallow_feat16 = self.stdc_net(inputs)
        # shallow_feat16=self.convert_shallow16(shallow_feat16)
        # output = self.cls_seg(shallow_feat16)

        # # stdc+freqfusion
        # shallow_feat8, shallow_feat16 = self.stdc_net(inputs)
        # shallow_feat16=self.convert_shallow16(shallow_feat16)
        # shallow_feat8=self.convert_shallow8(shallow_feat8)
        # _,_,h,w=shallow_feat16.size()
        # shallow_feat8=F.interpolate(shallow_feat8, size=(h*2, w*2), mode='bilinear', align_corners=False)
        # _,hr,lr=self.freqfusion(shallow_feat8,shallow_feat16)
        #
        # fusion=self.addConv(hr+lr)
        # output = self.cls_seg(fusion)

        # 无高频残差add
        shallow_feat8, shallow_feat16 = self.stdc_net(inputs)
        shallow_feat16=self.convert_shallow16(shallow_feat16)
        shallow_feat8=self.convert_shallow8(shallow_feat8)
        _, _, h, w = shallow_feat8.size()
        shallow_feat16 = F.interpolate(shallow_feat16, size=(h , w ), mode='bilinear', align_corners=False)
        fusion=self.addConv(shallow_feat8+shallow_feat16)
        output = self.cls_seg(fusion)



        if train_flag:
            if self.diff_flag==1:
                loss_shallow_diff, diff_map, diff_pred_shallow = self.shallow_diff.forward_train_diff(output, img_metas,
                                                                                                      gt, train_cfg)
                diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
                                           align_corners=self.align_corners)
                diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
                diff_pred_shallow_sig = diff_pred_shallow
                diff_pred_shallow = (diff_pred_shallow > 0.5).float()
                gt[diff_pred_shallow == 0] = 255
                return output, output_aux16,loss_shallow_diff
            elif self.diff_flag==2:
                return output, output_aux16
            elif self.diff_flag == 3:
                return output
            else:
                diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
                                   align_corners=self.align_corners)
                diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
                diff_pred_shallow_sig=diff_pred_shallow
                diff_pred_shallow = (diff_pred_shallow > 0.5).float()
                gt[diff_pred_shallow == 0] = 255

                output = resize(output, size=inputs.size()[2:], mode='bilinear',
                                   align_corners=self.align_corners)
                # diff_input=torch.concat([inputs,output],dim=1)
                # diff_count_of_ones = torch.sum(diff_pred_shallow == 1)
                diff_input=inputs*diff_pred_shallow
                b, c, h, w = diff_input.shape
                coords, features=get_coordsandfeatures(diff_input)
                diff_output = self.spnet(features, coords, b)
                diff_output=diff_output.dense()
                diff_concat=torch.concat([diff_output,output],dim=1)
                diff_fuse=diff_output
                # diff_fuse = self.sp_fuse(diff_concat)
                sp_loss = self.spnet.sp_loss(diff_fuse, diff_pred_shallow,img_metas,gt, train_cfg)
                mask = (diff_output != 0)
                output[mask] = diff_fuse[mask]
                last_output=output
                return output, output_aux16, loss_shallow_diff,sp_loss,last_output
        else:
            return output
            # else:
            #     diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
            #                                align_corners=self.align_corners)
            #     output = resize(output, size=inputs.size()[2:], mode='bilinear',
            #                     align_corners=self.align_corners)
            #     diff_pred_left, diff_pred_top_right, diff_pred_bottom_left, diff_pred_bottom_right = split_tensor(
            #         diff_pred_shallow)
            #     input_top_left, input_top_right, input_bottom_left, input_bottom_right = split_tensor(inputs)
            #     output_top_left, output_top_right, output_bottom_left, output_bottom_right = split_tensor(output)
            #     diff_pred_list = [diff_pred_left, diff_pred_top_right, diff_pred_bottom_left, diff_pred_bottom_right]
            #     inputs_list = [input_top_left, input_top_right, input_bottom_left, input_bottom_right]
            #     output_list = [ output_top_left, output_top_right, output_bottom_left, output_bottom_right]
            #     last_output_list=[]
            #     for i in range(4):
            #         diff_pred_shallow=diff_pred_list[i]
            #         inputs=inputs_list[i]
            #         output=output_list[i]
            #         diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
            #         diff_pred_shallow_sig = diff_pred_shallow
            #         diff_pred_shallow = (diff_pred_shallow > 0.5).float()
            #         # diff_input = torch.concat([inputs, output], dim=1)
            #         diff_input = inputs * diff_pred_shallow
            #         # diff_input=inputs*diff_pred_shallow
            #         b, c, h, w = diff_input.shape
            #         coords, features=get_coordsandfeatures(diff_input)
            #         n,c=coords.shape
            #         if n>0:
            #             diff_output = self.spnet(features, coords, b)
            #             diff_output=diff_output.dense()
            #             diff_concat = torch.concat([diff_output, output], dim=1)
            #             diff_fuse = diff_output
            #             # diff_fuse = self.sp_fuse(diff_concat)
            #             mask = (diff_output != 0)
            #             output[mask] = diff_fuse[mask]
            #
            #         last_output_list.append(output)
            #     last_output=restore_tensor(last_output_list[0],last_output_list[1],last_output_list[2],last_output_list[3])
            #
            #     return last_output


    # def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
    #     """Forward function."""
    #     prymaid_results = self.lap_prymaid_conv.pyramid_decom(inputs)
    #     high_residual_1 = prymaid_results[0]
    #     high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
    #                                     align_corners=False)
    #     high_residual_input = torch.cat([high_residual_1, high_residual_2], dim=1)
    #
    #     shallow_feat8, shallow_feat16,feat16_cls = self.stdc_net(high_residual_input)
    #     shallow_feat16=self.convert_shallow16(shallow_feat16)
    #     _, aux_feat8, fused_feat_8 = self.fuse8(shallow_feat8, shallow_feat16)
    #     output = self.cls_seg(fused_feat_8)
    #     output_aux16 = self.conv_seg_aux_16(aux_feat8)
    #
    #     if train_flag:
    #         loss_shallow_diff, diff_map, diff_pred_shallow = self.shallow_diff.forward_train_diff(output, img_metas,
    #                                                                                               gt, train_cfg)
    #
    #         if self.diff_flag==False:
    #             return output, output_aux16, loss_shallow_diff
    #         else:
    #             diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
    #                                align_corners=self.align_corners)
    #             diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
    #             diff_pred_shallow_sig = diff_pred_shallow
    #             diff_pred_shallow = (diff_pred_shallow > 0.5).float()
    #             output = resize(output, size=inputs.size()[2:], mode='bilinear',
    #                                align_corners=self.align_corners)
    #
    #             diff_output = self.stdc_net_diff(high_residual_input)
    #             diff_output = resize(diff_output, size=inputs.size()[2:], mode='bilinear',
    #                                align_corners=self.align_corners)
    #             diff_concat=torch.concat([diff_output,output],dim=1)
    #             diff_fuse=diff_output
    #             sp_loss = self.spnet.sp_loss(diff_fuse, diff_pred_shallow,img_metas,gt, train_cfg)
    #             diff_fuse = self.bag(diff_output,output,diff_pred_shallow_sig)
    #             # mask = (diff_pred_shallow != 0)
    #             # mask = torch.cat([mask, mask], dim=1)
    #             # output[mask] = diff_fuse[mask]
    #             # last_output=output
    #             return output, output_aux16, loss_shallow_diff,sp_loss,diff_fuse
    #     else:
    #         diff_pred_shallow=self.shallow_diff.forward_test_diff(output, img_metas)
    #         if self.diff_flag==False:
    #             return output
    #         else:
    #             diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
    #                                align_corners=self.align_corners)
    #             diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
    #             diff_pred_shallow_sig = diff_pred_shallow
    #             diff_pred_shallow = (diff_pred_shallow > 0.5).float()
    #             output = resize(output, size=inputs.size()[2:], mode='bilinear',
    #                                align_corners=self.align_corners)
    #
    #             diff_output = self.stdc_net_diff(high_residual_input)
    #             diff_output = resize(diff_output, size=inputs.size()[2:], mode='bilinear',
    #                                align_corners=self.align_corners)
    #             # diff_concat=torch.concat([diff_output,output],dim=1)
    #             diff_fuse=diff_output
    #             diff_fuse = self.bag(diff_output,output,diff_pred_shallow_sig)
    #             # mask = (diff_pred_shallow != 0)
    #             # mask = torch.cat([mask, mask], dim=1)
    #             # output[mask] = diff_fuse[mask]
    #
    #
    #             return diff_fuse




    # def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
    #     """Forward function."""
    #     prymaid_results = self.lap_prymaid_conv.pyramid_decom(inputs)
    #     high_residual_1 = prymaid_results[0]
    #     high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
    #                                     align_corners=False)
    #     high_residual_input = torch.cat([high_residual_1, high_residual_2], dim=1)
    #
    #     shallow_feat8, shallow_feat16,feat16_cls = self.stdc_net(high_residual_input)
    #     shallow_feat16=self.convert_shallow16(shallow_feat16)
    #     _, aux_feat8, fused_feat_8 = self.fuse8(shallow_feat8, shallow_feat16)
    #     output = self.cls_seg(fused_feat_8)
    #     output_aux16 = self.conv_seg_aux_16(aux_feat8)
    #
    #     if train_flag:
    #         loss_shallow_diff, diff_map, diff_pred_shallow = self.shallow_diff.forward_train_diff(output, img_metas,
    #                                                                                               gt, train_cfg)
    #
    #         if self.diff_flag==False:
    #             return output, output_aux16, loss_shallow_diff
    #         else:
    #             diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
    #                                align_corners=self.align_corners)
    #             diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
    #             diff_pred_shallow_sig = diff_pred_shallow
    #             diff_pred_shallow = (diff_pred_shallow > 0.5).float()
    #             output = resize(output, size=inputs.size()[2:], mode='bilinear',
    #                                align_corners=self.align_corners)
    #
    #             diff_output = self.stdc_net_diff(high_residual_input)
    #             diff_output = resize(diff_output, size=inputs.size()[2:], mode='bilinear',
    #                                align_corners=self.align_corners)
    #             diff_concat=torch.concat([diff_output,output],dim=1)
    #             diff_fuse=diff_output
    #             sp_loss = self.spnet.sp_loss(diff_fuse, diff_pred_shallow,img_metas,gt, train_cfg)
    #             diff_fuse = self.bag(diff_output,output,diff_pred_shallow_sig)
    #             # mask = (diff_pred_shallow != 0)
    #             # mask = torch.cat([mask, mask], dim=1)
    #             # output[mask] = diff_fuse[mask]
    #             # last_output=output
    #             return output, output_aux16, loss_shallow_diff,sp_loss,diff_fuse
    #     else:
    #         diff_pred_shallow=self.shallow_diff.forward_test_diff(output, img_metas)
    #         if self.diff_flag==False:
    #             return output
    #         else:
    #             diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
    #                                align_corners=self.align_corners)
    #             diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
    #             diff_pred_shallow_sig = diff_pred_shallow
    #             diff_pred_shallow = (diff_pred_shallow > 0.5).float()
    #             output = resize(output, size=inputs.size()[2:], mode='bilinear',
    #                                align_corners=self.align_corners)
    #
    #             diff_output = self.stdc_net_diff(high_residual_input)
    #             diff_output = resize(diff_output, size=inputs.size()[2:], mode='bilinear',
    #                                align_corners=self.align_corners)
    #             # diff_concat=torch.concat([diff_output,output],dim=1)
    #             diff_fuse=diff_output
    #             diff_fuse = self.bag(diff_output,output,diff_pred_shallow_sig)
    #             # mask = (diff_pred_shallow != 0)
    #             # mask = torch.cat([mask, mask], dim=1)
    #             # output[mask] = diff_fuse[mask]
    #
    #
    #             return diff_fuse



    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg,mask=None):
        if self.diff_flag==1:
            seg_logits, seg_logits_aux16,loss_shallow_diff= self.forward(inputs, prev_output,
                                                                                                 mask=mask,
                                                                                                 gt=gt_semantic_seg,
                                                                                                 img_metas=img_metas,
                                                                                                 train_cfg=train_cfg,
                                                                                                 diff_pred_deep=mask)
            losses = self.losses(seg_logits, gt_semantic_seg)
            losses_aux16 = self.losses(seg_logits_aux16, gt_semantic_seg)
            losses['loss_seg']=losses['loss_seg']*2
            losses_aux16['loss_seg']=losses_aux16['loss_seg']

            # losses_aux16 = torch.tensor(0)

            # last_losses = self.losses(last_output, gt_semantic_seg)

            # losses_aux8 = self.losses(seg_logits_aux8, gt_semantic_seg)
            # losses_detail = self.detail_loss(shallow_feat4_cls,gt_semantic_seg.squeeze(1) )
            # return losses, losses_aux16, loss_shallow_diff
            return losses, losses_aux16,\
                   loss_shallow_diff


        elif self.diff_flag==2:
            seg_logits, seg_logits_aux16= self.forward(inputs, prev_output,
                                                                                                 mask=mask,
                                                                                                 gt=gt_semantic_seg,
                                                                                                 img_metas=img_metas,
                                                                                                 train_cfg=train_cfg,
                                                                                                 diff_pred_deep=mask)
            losses = self.losses(seg_logits, gt_semantic_seg)
            losses_aux16 = self.losses(seg_logits_aux16, gt_semantic_seg)
            # losses['loss_seg']=losses['loss_seg']*2
            # losses_aux16['loss_seg']=losses_aux16['loss_seg']

            # losses_aux16 = torch.tensor(0)

            # last_losses = self.losses(last_output, gt_semantic_seg)

            # losses_aux8 = self.losses(seg_logits_aux8, gt_semantic_seg)
            # losses_detail = self.detail_loss(shallow_feat4_cls,gt_semantic_seg.squeeze(1) )
            # return losses, losses_aux16, loss_shallow_diff
            return losses, losses_aux16

        elif self.diff_flag==3:
            seg_logits= self.forward(inputs, prev_output,
                                                                                                 mask=mask,
                                                                                                 gt=gt_semantic_seg,
                                                                                                 img_metas=img_metas,
                                                                                                 train_cfg=train_cfg,
                                                                                                 diff_pred_deep=mask)
            losses = self.losses(seg_logits, gt_semantic_seg)
            # loss_aux=dict()
            # loss_aux=losses
            # losses_aux['loss_seg'] = 0
            return losses

        else:

            seg_logits, seg_logits_aux16,loss_shallow_diff,sp_loss,last_output= self.forward(inputs, prev_output,mask=mask,gt=gt_semantic_seg,img_metas=img_metas,train_cfg=train_cfg,diff_pred_deep=mask)
            losses = self.losses(seg_logits, gt_semantic_seg)
            losses_aux16 = self.losses(seg_logits_aux16, gt_semantic_seg)
            loss_fuse=self.losses(last_output,gt_semantic_seg)
            # last_losses = self.losses(last_output, gt_semantic_seg)

            # losses_aux8 = self.losses(seg_logits_aux8, gt_semantic_seg)
            # losses_detail = self.detail_loss(shallow_feat4_cls,gt_semantic_seg.squeeze(1) )
            return losses, losses_aux16, loss_shallow_diff,sp_loss,loss_fuse

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
