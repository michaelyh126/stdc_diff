#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 用标准 BatchNorm2d 代替你原代码中的 InPlaceABNSync
# 原代码依赖:
# from modules.bn import InPlaceABNSync as BatchNorm2d
# =========================================================
BatchNorm2d = nn.BatchNorm2d


# =========================================================
# ResNet18 Backbone
# 这里补上你原代码缺失的:
# from resnet import Resnet18
#
# 输出:
# feat8  : 1/8
# feat16 : 1/16
# feat32 : 1/32
# 通道分别为:
# 128, 256, 512
# 这与你原始 ContextPath 里的 ARM 输入通道匹配
# =========================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chan, out_chan, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_chan, out_chan, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_chan, out_chan, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(out_chan)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )

        self.layer1 = self._make_layer(64, 2, stride=1)   # 1/4
        self.layer2 = self._make_layer(128, 2, stride=2)  # 1/8
        self.layer3 = self._make_layer(256, 2, stride=2)  # 1/16
        self.layer4 = self._make_layer(512, 2, stride=2)  # 1/32

        self.init_weight()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                BatchNorm2d(planes)
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 1/4

        x = self.layer1(x)  # 1/4
        feat8 = self.layer2(x)   # 1/8,  128 channels
        feat16 = self.layer3(feat8)  # 1/16, 256 channels
        feat32 = self.layer4(feat16) # 1/32, 512 channels

        return feat8, feat16, feat32

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# =========================================================
# 原始 BiSeNet 模块
# =========================================================
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for _, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.adaptive_avg_pool2d(feat, output_size=1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


import torch
import torch.nn as nn
import torch.nn.functional as F


class CP(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CP, self).__init__()
        self.resnet = Resnet18()

        # --- 替换 ARM 为普通卷积 ---
        # 假设 resnet 输出的 feat16 通道数为 256，feat32 通道数为 512
        # 使用 3x3 卷积，输出通道数强制设为 128
        self.conv16 = ConvBNReLU(256, 128, ks=3, stride=1, padding=1)
        self.conv32 = ConvBNReLU(512, 128, ks=3, stride=1, padding=1)

        # 去掉了 self.conv_head32 和 self.conv_head16，因为前面的普通卷积已经输出128通道了
        # 去掉了 self.conv_avg，因为你上一版代码中已经没有使用全局平均池化分支了

        # 融合卷积
        # 假设 feat8 是 64 通道，加上两个 128 通道，总共是 64 + 128 + 128 = 320
        # 如果你确认 feat8 也是 128 通道，那这里就是 128 * 3 = 384
        self.fusion_conv = ConvBNReLU(128 * 3, 128, ks=3, stride=1, padding=1)
        # 注意：上面这行请务必根据实际 feat8 的通道数修改，如果是64通道，请改为 320

        self.init_weight()

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]

        # 1. 使用普通卷积处理 feat32 并上采样
        feat32_out = self.conv32(feat32)
        feat32_up = F.interpolate(feat32_out, (H8, W8), mode='nearest')

        # 2. 使用普通卷积处理 feat16 并上采样
        feat16_out = self.conv16(feat16)
        feat16_up = F.interpolate(feat16_out, (H8, W8), mode='nearest')

        # 3. 融合 feat8, feat16_up, feat32_up
        fused_feat = torch.cat([feat8, feat16_up, feat32_up], dim=1)
        output = self.fusion_conv(fused_feat)

        return output

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for _, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.adaptive_avg_pool2d(feat32, output_size=1)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up   # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for _, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for _, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


import torch
import torch.nn as nn
import torch.nn.functional as F


class SP(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SP, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)

        # 新增：用于融合拼接后的特征 (64 + 64 + 64 = 192) -> 128
        # 如果你的后续网络不需要通道数对齐，可以把这行删掉
        self.fusion_conv = ConvBNReLU(192, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        # feat1: 1/2 尺寸, 64通道
        feat1 = self.conv1(x)
        H, W = feat1.size()[2:]

        # feat2: 1/4 尺寸, 64通道
        feat2 = self.conv2(feat1)

        # feat3: 1/8 尺寸, 64通道
        feat3 = self.conv3(feat2)

        # 将 feat2 和 feat3 都上采样到 feat1 的尺寸 (1/2)
        feat2_up = F.interpolate(feat2, size=(H, W), mode='nearest')
        feat3_up = F.interpolate(feat3, size=(H, W), mode='nearest')

        # 在通道维度拼接: [B, 64+64+64, H/2, W/2]
        fused_feat = torch.cat([feat1, feat2_up, feat3_up], dim=1)

        # 通道融合降维
        out = self.fusion_conv(fused_feat)

        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for _, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(
            out_chan,
            out_chan // 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            out_chan // 4,
            out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)

        atten = F.adaptive_avg_pool2d(feat, output_size=1)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)

        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for _, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)

        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]

        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)

        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        lr_mul_wd_params, lr_mul_nowd_params = [], []

        for _, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params

        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


# =========================================================
# test
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # net = BiSeNet(n_classes=19).to(device)
    # net.eval()
    net = CP().to(device)
    net.eval()
    in_ten = torch.randn(2, 3, 640, 480).to(device)

    with torch.no_grad():
        out, out16, out32 = net(in_ten)

    print("input :", in_ten.shape)
    print("out   :", out.shape)
    print("out16 :", out16.shape)
    print("out32 :", out32.shape)

    net.get_params()
    print("forward success.")
