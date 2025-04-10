# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .pid import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag,AddFuse,HarrFusion,ConcatFuse,UncertaintyFusion,AddFuse_sample
from .diff_head import DiffHead
import logging
import math
from mmcv.runner import BaseModule
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,trunc_normal_init,normal_init)
from timm.models.layers import DropPath
from ..builder import BACKBONES
from mmcv.cnn import (Conv2d,ConvModule)

# from .diff_fusion import FFM

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

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


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, sync=False):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        if sync:
            self.bn = nn.SyncBatchNorm(out_planes)
        else:
            self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


# From https://github.com/MichaelFan01/STDC-Seg
class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


# From https://github.com/MichaelFan01/STDC-Seg
class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


class PIDNet(nn.Module):

    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet, self).__init__()
        self.augment = augment
        # self.a = nn.Parameter(torch.zeros(1))
        self.fuse_shallow = RelationAwareFusion(planes*4, None, dict(type='BN', requires_grad=True), dict(type='ReLU'), ext=2)
        self.cls8= nn.Sequential(
                          nn.Conv2d(64,1,kernel_size=1, stride=1),
                          )
        self.add_fuse = AddFuse(planes * 16, planes * 4)
        self.cat_fuse=ConcatFuse(planes*8,planes*4)
        self.cat_shallow=ConcatFuse(planes*12,planes*4)
        self.p_add3=AddFuse_sample(planes*4,planes*4)
        self.p_add4=AddFuse_sample(planes*8,planes*8)
        self.i_add3=AddFuse_sample(planes*4,planes*4)
        self.i_add4=AddFuse_sample(planes*8,planes*8)

        # self.cfb1 = CFBlock(
        #     in_channels=planes * 8,
        #     out_channels=planes * 8,
        #     num_heads=8,
        #     drop_rate=0.,
        #     drop_path_rate=0.1)
        # self.cfb2 = CFBlock(
        #     in_channels=planes * 16,
        #     out_channels=planes * 16,
        #     num_heads=8,
        #     drop_rate=0.,
        #     drop_path_rate=0.1)
        # self.ffm=FFM(planes*4,planes*4)

        # I Branch
        self.conv1 =  nn.Sequential(
                          nn.Conv2d(6,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, planes*2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(planes*2, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # P Branch
        self.layers_=self._make_layers(64,[2,2,2],4,CatBottleneck)
        self.layer3_ = nn.Sequential(self.layers_[2:4])
        self.layer4_ = nn.Sequential(self.layers_[4:6])
        self.layer5_ = nn.Sequential(self.layers_[6:8])
        self.compression = nn.Sequential(
                                          nn.Conv2d(planes * 16, planes * 4, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 4, momentum=bn_mom),
                                          )
        #
        # self.compression4 = nn.Sequential(
        #                                   nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
        #                                   BatchNorm2d(planes * 2, momentum=bn_mom),
        #                                   )
        # self.pag3 = PagFM(planes * 2, planes)
        # self.pag4 = PagFM(planes * 2, planes)
        # self.uf3 = UncertaintyFusion(planes*2)
        # self.uf4 = UncertaintyFusion(planes*2)

        # self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        # self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        # self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # D Branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                                        nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                                        BatchNorm2d(planes * 2, momentum=bn_mom),
                                        )
            self.diff4 = nn.Sequential(
                                     nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                                     BatchNorm2d(planes * 2, momentum=bn_mom),
                                     )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)


        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # Prediction Head
        if self.augment:
            self.seghead_p = segmenthead(planes * 4, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 2, planes, 1)

        self.final_layer = segmenthead(planes * 4, head_planes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(6, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*2, block_num, 1))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i)), base*int(math.pow(2,i+1)), block_num, 1))
                else:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+1)), block_num, 1))

        return nn.Sequential(*features)

    def _make_stdc(self, base, layers, block_num, block):
        features = []
        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*2, block_num, 1))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i)), base*int(math.pow(2,i+1)), block_num, 1))
                else:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+1)), block_num, 1))

        return nn.Sequential(*features)


    def forward_t(self, x):

        width_output = math.ceil(x.shape[-1] / 8)
        height_output = math.ceil(x.shape[-2] / 8)
        feats=[]
        feats.append(x)


        x = self.conv1(x)

        x=self.conv2(x)
        x = self.layer1(x)


        x=self.layer2(self.relu(x))
        x = self.relu(x)

        # x_=self.conv3(x_)
        x_ = self.layer3_(x)
        x8=x_

        x=self.layer3(x)
        feats.append(x)

        x = self.relu(x)
        x=self.i_add3(x,x_)
        x_=self.p_add3(x_,x)


        if self.augment:
            temp_p = x_

        x=self.layer4(x)
        feats.append(x)
        x = self.relu(x)
        x_ = self.layer4_(x_)


        x=self.i_add4(x,x_)
        x_=self.p_add4(x_,x)


        x_ = self.layer5_(x_)
        # x_=self.cat_shallow(x8,x_)
        x=self.layer5(x)
        feats.append(x)


        x = F.interpolate(
            self.spp(x),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc)

        feats.append(x)

        x_=self.compression(x_)
        x16=x_
        # _, aux_feat16, x_ = self.fuse_shallow(x8, x16)
        # x_=F.interpolate(
        #     x_,
        #     size=[height_output, width_output],
        #     mode='bilinear', align_corners=algc)
        # a=torch.sigmoid(self.a)
        fusion=self.cat_fuse(x_,x)
        # fusion=self.dfm(x_, x, x8_cls)
        x_ = self.final_layer(fusion)


        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            return [x_extra_p, x_ ,feats,fusion]
        else:
            return x_


    def forward_shallow(self, x):

        width_output = math.ceil(x.shape[-1] / 8)
        height_output = math.ceil(x.shape[-2] / 8)
        feats=[]
        feats.append(x)


        x = self.conv1(x)

        x=self.conv2(x)
        x = self.layer1(x)
        x=self.layer2(self.relu(x))
        x = self.relu(x)

        # x_=self.conv3(x_)
        x_ = self.layer3_(x)
        x8=x_
        if self.augment:
            temp_p = x_


        x_ = self.layer4_(x_)
        x_ = self.layer5_(x_)
        x_=self.compression(x_)
        x16=x_
        x_ = self.final_layer(x_)


        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            return [x_extra_p, x_ ,feats,0]
        else:
            return x_

