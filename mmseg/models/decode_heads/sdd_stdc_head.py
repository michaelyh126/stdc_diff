import torch
import torch.nn as nn
from torch.nn import init
import math
import time
from mmcv.cnn import ConvModule
from collections import OrderedDict
from mmseg.ops import resize
# from .diff_fusion import FeatureFusionModule
from .pid import Bag,PagFM

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
        self.init_params()
        # if pretrain_model:
        #     print('use pretrain model {}'.format(pretrain_model))
        #     self.init_weight(pretrain_model)
        # else:
        #     self.init_params()

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
        # feat16_cls=self.cls_feat16(feat16)
        if cas3:
            return feat4, feat8, feat16
        else:
            return feat8, feat16


class ShallowNet_diff(nn.Module):
    def __init__(self, base=64, in_channels=3,  layers=[2,2], block_num=4, type="cat", dropout=0.20, pretrain_model="D:\isd\pretrained models\STDCNet813M_73.91.tar",num_classes=2):
        super(ShallowNet_diff, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.in_channels = in_channels
        self.cls_feat16=ConvX(512,num_classes,3,1)
        self.cls_feat8=ConvX(256,num_classes,3,1)
        self.features = self._make_layers(base, layers, block_num, block)
        # self.fusion1=FeatureFusionModule()
        # self.pag2=PagFM(32,8,with_channel=True).cuda()
        # self.pag4=PagFM(64,16,with_channel=True).cuda()
        # self.pag8=PagFM(256,64,with_channel=True).cuda()
        # self.pag16=PagFM(512,128,with_channel=True).cuda()
        # self.bag=Bag(512,128)
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
        # feat16 = self.x16(feat8)
        feat8_cls=self.cls_feat8(feat8)
        if cas3:
            return feat4, feat8, feat16
        else:
            return feat8_cls


class ShallowNet_RF63(nn.Module):
    """
    单分支 STDC 版本：
    - stem: x2, x4
    - 第一个下采样 block 到 x8，作为“目标感受野锚点”
    - 之后所有 block 都不再 stride=2
    - 继续用 stride=1 的 STDC block 做细化

    结构：
        x2  -> ConvX(3x3, s=2)
        x4  -> ConvX(3x3, s=2)
        x8_anchor -> block(base, base*4, stride=2)   # 这里 RF ~ 63
        x8_refine1 -> block(256, 256, stride=1)
        x8_refine2 -> block(256, 512, stride=1)      # 只升通道，不降采样
        x8_refine3 -> block(512, 512, stride=1)

    返回：
        feat8_anchor: 第一个 x8 block 输出（RF 锚点）
        feat8_deep  : 更深的 x8 特征（仍是 x8 分辨率）
    """
    def __init__(
        self,
        base=64,
        in_channels=3,
        block_num=4,
        type="cat",
        num_classes=2
    ):
        super(ShallowNet_RF63, self).__init__()

        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        else:
            raise ValueError("type must be 'cat' or 'add'")

        self.in_channels = in_channels

        # stem
        self.x2 = ConvX(in_channels, base // 2, 3, 2)   # x2
        self.x4 = ConvX(base // 2, base, 3, 2)          # x4

        # 第一个 x8 block：保留原始 stride=2
        # 这是你要的“目标感受野锚点”，理论最大 RF 约为 63
        self.stage1_down = block(base, base * 4, block_num, stride=2)   # x8, 256 ch

        # 后面全部 stride=1，不再继续池化/下采样
        self.stage1_refine = block(base * 4, base * 4, block_num, stride=1)   # x8, 256 ch
        self.stage2_expand = block(base * 4, base * 8, block_num, stride=1)   # x8, 512 ch
        self.stage2_refine = block(base * 8, base * 8, block_num, stride=1)   # x8, 512 ch

        # 可选分类头，方便你调试
        self.cls_feat8_anchor = ConvX(base * 4, num_classes, 3, 1)
        self.cls_feat8_deep = ConvX(base * 8, num_classes, 3, 1)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, cas3=False, return_logits=False):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)

        # RF anchor ~ 63
        feat8_anchor = self.stage1_down(feat4)

        # 后续仍为 x8，只做细化
        feat8_mid = self.stage1_refine(feat8_anchor)
        feat8_deep = self.stage2_refine(self.stage2_expand(feat8_mid))

        if return_logits:
            logit_anchor = self.cls_feat8_anchor(feat8_anchor)
            logit_deep = self.cls_feat8_deep(feat8_deep)
            return logit_anchor, logit_deep

        if cas3:
            # 为了和你之前风格接近，返回 feat4、x8锚点、深层x8
            return feat4, feat8_anchor, feat8_deep
        else:
            # 类似原来返回 feat8, feat16
            # 这里只不过第二个不是 x16，而是更深的 x8
            return feat8_anchor, feat8_deep
