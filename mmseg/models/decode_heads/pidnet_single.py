# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .pid import BasicBlock, ConvNextBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag,AddFuse,ConcatFuse,AdaptiveFrequencyFusion,CBAMLayer
from .diff_head import DiffHead
import logging
from .isdhead import RelationAwareFusion

import math
from mmseg.models.sampler.dysample import DySample

# from .diff_fusion import FFM

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


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


class Uncertainty_Rank_Algorithm(nn.Module):
    def __init__(self):
        super(Uncertainty_Rank_Algorithm, self).__init__()
        # self.prob = nn.Sigmoid()

    def forward(self, map):
        prob_map = map
        fore_uncertainty_map = prob_map - 0.5
        back_uncertainty_map = 0.5 - prob_map

        fore_rank_map = torch.zeros_like(map)
        back_rank_map = torch.zeros_like(map)

        fore_rank_map[fore_uncertainty_map > 0.] = 5
        fore_rank_map[fore_uncertainty_map > 0.1] = 4
        fore_rank_map[fore_uncertainty_map > 0.2] = 3
        fore_rank_map[fore_uncertainty_map > 0.3] = 2
        fore_rank_map[fore_uncertainty_map > 0.4] = 1

        back_rank_map[back_uncertainty_map > 0.] = 5
        back_rank_map[back_uncertainty_map > 0.1] = 4
        back_rank_map[back_uncertainty_map > 0.2] = 3
        back_rank_map[back_uncertainty_map > 0.3] = 2
        back_rank_map[back_uncertainty_map > 0.4] = 1

        return fore_rank_map.detach(), back_rank_map.detach()


class Uncertainty_Aware_Fusion_Module(nn.Module):
    def __init__(self, high_channel, out_channel):
        super(Uncertainty_Aware_Fusion_Module, self).__init__()
        self.rank = Uncertainty_Rank_Algorithm()
        self.high_channel = high_channel
        self.out_channel = out_channel
        self.conv_high = BasicConv2d(self.high_channel, self.out_channel, 3, 1, 1)
        self.conv_fusion = nn.Conv2d(2 * self.out_channel, self.out_channel, 3, 1, 1)

        # self.seg_out = nn.Conv2d(self.out_channel, num_classes, 1)

    def forward(self, feature_high, map):
        map = map.unsqueeze(1)
        # uncertainty_fore_map_high, uncertainty_back_map_high = self.rank(map)
        # uncertainty_feature_high = torch.cat(
        #     (uncertainty_fore_map_high * feature_high, uncertainty_back_map_high * feature_high), dim=1)
        # seg_fusion=self.conv_high(feature_high*map+feature_high)
        seg_fusion=self.conv_high(feature_high)
        # seg = self.seg_out(seg_fusion)
        return seg_fusion

def get_mask(map):
    topk_values, topk_indices = torch.topk(map, 2, dim=1, largest=True, sorted=False)
    mask = torch.abs(topk_values[:, 0, :, :] - topk_values[:, 1, :, :])
    mask=torch.exp(-mask)*5
    return mask

class PIDNet(nn.Module):

    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet, self).__init__()
        self.augment = augment
        # self.cbam=CBAMLayer(planes*4)
        self.nopappm= nn.Sequential(
                          nn.Conv2d(512,128,kernel_size=3, stride=1, padding=1),
                          BatchNorm2d(128, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )
        self.cat_fuse = ConcatFuse(planes * 8, planes * 4)
        self.add_fuse = AddFuse(planes*4,planes*4)
        self.aff = AdaptiveFrequencyFusion(sp_channels=128, co_channels=128, out_channels=128, mid_channels=128,
                                           kernel_size=3)
        self.raf=RelationAwareFusion(planes*4, None, dict(type='BN', requires_grad=True), dict(type='ReLU'), ext=1)
        self.dysample=DySample(128,scale=8,groups=4)
        self.unlayer3=Uncertainty_Aware_Fusion_Module(planes*4,planes*4)
        self.unlayer4=Uncertainty_Aware_Fusion_Module(planes*4,planes*4)
        self.unlayer5=Uncertainty_Aware_Fusion_Module(planes*4,planes*4)
        # self.ffm=FFM(planes*4,planes*4)

        # I Branch
        self.conv1 =  nn.Sequential(
                          nn.Conv2d(6,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 4, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # P Branch
        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 2, momentum=bn_mom),
                                          )

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
                                          BatchNorm2d(planes * 2, momentum=bn_mom),
                                          )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

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
            self.seghead_p = segmenthead(planes * 2, head_planes, num_classes)
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

        return layer


    def forward(self, x):

        x_ori=x
        width_output = x.shape[-1]
        height_output = x.shape[-2]


        x = self.relu(self.layer3(x))
        # x_ = self.pag3(x_, self.compression3(x))
        # if self.augment:
        #     temp_p = x_

        x = self.relu(self.layer4(x))
        # x_ = self.pag4(x_, self.compression4(x))


        # x_ = self.layer5_(self.relu(x_))
        x=self.layer5(x)
        # x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(
            self.spp(x),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc)
        fusion=self.cat_fuse(x_ori, x)
        x_ = self.final_layer(fusion)
        # x_ = self.final_layer(x_)

        if self.augment:
            # x_extra_p = self.seghead_p(temp_p)
            # x_extra_d = self.seghead_d(temp_d)
            return x_
        else:
            return x_

    def forward_sp(self, x,map):
        x_ori=x
        width_output = x.shape[-1]
        height_output = x.shape[-2]
        mask=get_mask(map)
        x=self.unlayer3(x,mask)
        x=self.unlayer4(x,mask)
        x=self.unlayer5(x,mask)
        fusion = self.add_fuse(x_ori, x)

        x_ = self.final_layer(fusion)
        if self.augment:
            # x_extra_p = self.seghead_p(temp_p)
            # x_extra_d = self.seghead_d(temp_d)
            return x_
        else:
            return x_

    def forward_dual(self, x,map):
        x_=x
        x_ori=x
        width_output = x.shape[-1]
        height_output = x.shape[-2]
        # mask=get_mask(map)
        # x_=self.unlayer3(x_,mask)
        # x_=self.unlayer4(x_,mask)
        # x_=self.unlayer5(x_,mask)
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x=self.layer5(x)

        # x=self.spp(x)
        # x=self.dysample(x)
        # x = F.interpolate(
        #     x,
        #     size=[height_output, width_output],
        #     mode='bilinear', align_corners=algc)

        # PAPPM
        x = F.interpolate(
            self.spp(x),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc)

        # no PAPPM
        # x = F.interpolate(
        #     self.nopappm(x),
        #     size=[height_output, width_output],
        #     mode='bilinear', align_corners=algc)


        fusion = self.add_fuse(x_ori, x)
        # fusion=self.cat_fuse(x_ori,x)
        # fusion=self.aff(x_ori,x)
        # _,_,fusion=self.raf(x_ori,x)

        # fusion = self.cbam(fusion)
        x_ = self.final_layer(fusion)
        if self.augment:
            # x_extra_p = self.seghead_p(temp_p)
            # x_extra_d = self.seghead_d(temp_d)
            return x_
        else:
            return x_

def get_seg_model(cfg, imgnet_pretrained):

    if 's' in cfg.MODEL.NAME:
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, ppm_planes=96, head_planes=128, augment=True)
    elif 'm' in cfg.MODEL.NAME:
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=96, head_planes=128, augment=True)
    else:
        model = PIDNet(m=3, n=4, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=112, head_planes=256, augment=True)

    if imgnet_pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')['state_dict']
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict = False)
    else:
        pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict = False)

    return model

def get_pred_model(name, num_classes):

    if 's' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif 'm' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)
    else:
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False)

    return model

if __name__ == '__main__':

    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    model = get_pred_model(name='pidnet_s', num_classes=19)
    model.eval()
    model.to(device)
    iterations = None

    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)





