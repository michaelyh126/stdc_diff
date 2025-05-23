import torch
import torch.nn as nn
from torch.nn import init
import math
import time
from mmcv.cnn import ConvModule
from collections import OrderedDict
from mmseg.ops import resize
from .diff_head import DiffHead
from .pid import Bag,PagFM
from .pid import BasicBlock

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

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

class ShallowNet(nn.Module):
    def __init__(self, base=64, in_channels=3,  layers=[2,2], block_num=4, type="cat", dropout=0.20, pretrain_model="D:\isd\pretrained models\STDCNet813M_73.91.tar",num_classes=2):
        super(ShallowNet, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.in_channels = in_channels
        self.cls_feat16=ConvX(512,num_classes,3,1)
        self.features = self._make_layers(base, layers, block_num, block)
        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if k == 'features.0.conv.weight' and self.in_channels != 3:
                v = torch.cat([v, v], dim=1)
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict, strict=False)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(self.in_channels, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x, cas3=False):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        # feat4_cls = self.cls_feat4(feat4)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat16_cls=self.cls_feat16(feat16)
        if cas3:
            return feat4, feat8, feat16
        else:
            return feat8, feat16,feat16_cls



class ShallowNet_diff(nn.Module):
    def __init__(self, base=64, in_channels=3,  layers=[2,2,2], block_num=4, type="cat", dropout=0.20, pretrain_model="D:\isd\pretrained models\STDCNet813M_73.91.tar",num_classes=2):
        super(ShallowNet_diff, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.in_channels = in_channels
        self.cls_feat16=ConvX(512,num_classes,3,1)
        self.cls_feat8=ConvX(256,num_classes,3,1)
        self.features = self._make_layers(base, layers, block_num, block)
        self.shallow_diff8=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=num_classes,align_corners=False,loss_decode=dict(type='BCEDiceLoss'))
        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        # self.x32 = nn.Sequential(self.features[6:8])
        self.x8_1 = self._make_layer(BasicBlock, 256, 512, 2, stride=1)
        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if k == 'features.0.conv.weight' and self.in_channels != 3:
                v = torch.cat([v, v], dim=1)
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict, strict=False)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(self.in_channels, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

        return nn.Sequential(*features)

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

    def forward(self, x, cas3=False):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat8_cls=self.cls_feat8(feat8)
        feat8_1=self.x8_1(feat8)
        # loss_shallow_diff8, _, diff_pred_shallow8 = self.shallow_diff8.forward_train_diff(feat8_cls, img_metas,
        #                                                                                       gt, train_cfg)
        # diff_pred_shallow8 = resize(diff_pred_shallow8, size=inputs.size()[2:], mode='bilinear',
        #                            align_corners=self.align_corners)
        # diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
        # diff_pred_shallow_sig = diff_pred_shallow
        # diff_pred_shallow = (diff_pred_shallow > 0.5).float()
        feat16 = self.x16(feat8)
        # feat32=self.x32(feat16)

        if cas3:
            return feat4, feat8, feat16
        else:
            return feat8,feat8_1,feat16,feat8_cls


    def forward_test(self, x, img_metas, gt, train_cfg, cas3=False):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        diff_pred_shallow = self.shallow_diff8.forward_test_diff(output, img_metas,gt, train_cfg)
        feat16 = self.x16(feat8)
        feat32=self.x32(feat16)

        if cas3:
            return feat4, feat8, feat16
        else:
            return feat2,feat4,feat8,feat16,feat32
