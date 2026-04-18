#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 用标准 BatchNorm2d
# =========================================================
BatchNorm2d = nn.BatchNorm2d


# =========================================================
# 基础卷积模块
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


# =========================================================
# ResNet18 BasicBlock
# 支持 dilation
# =========================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chan, out_chan, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn1 = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_chan,
            out_chan,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False
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


# =========================================================
# 改造版 ResNet18
# 关键要求：
# 后三层不再下采样
#
# 原始 ResNet18:
# conv1(stride=2) -> maxpool(stride=2) -> layer1 -> layer2(s2) -> layer3(s2) -> layer4(s2)
#
# 现在改成:
# conv1(stride=2) -> maxpool(stride=2) -> layer1(stride=1) -> layer2(stride=1)
# -> layer3(stride=1) -> layer4(stride=1)
#
# 所以 feat8 / feat16 / feat32 实际尺寸都相同，均为输入的 1/4
# 这里只是沿用原命名，方便接口兼容
# =========================================================
class Resnet18NoDownsample(nn.Module):
    def __init__(self, use_dilation=False):
        super(Resnet18NoDownsample, self).__init__()
        self.inplanes = 64
        self.use_dilation = use_dilation

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )

        # layer1 保持 1/4
        self.layer1 = self._make_layer(64, 2, stride=1, dilation=1)

        # 后三层都不再下采样
        # 可选 dilation 补一点感受野
        d2 = 1
        d3 = 2 if use_dilation else 1
        d4 = 4 if use_dilation else 1

        self.layer2 = self._make_layer(128, 2, stride=1, dilation=d2)  # still 1/4
        self.layer3 = self._make_layer(256, 2, stride=1, dilation=d3)  # still 1/4
        self.layer4 = self._make_layer(512, 2, stride=1, dilation=d4)  # still 1/4

        self.init_weight()

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                BatchNorm2d(planes)
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, stride=1, downsample=None, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 1/4

        x = self.layer1(x)       # 1/4, 64 channels
        feat8 = self.layer2(x)   # 1/4, 128 channels
        feat16 = self.layer3(feat8)  # 1/4, 256 channels
        feat32 = self.layer4(feat16) # 1/4, 512 channels

        return feat8, feat16, feat32

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# =========================================================
# 类似你原始代码里的 CP
# 但由于 feat8/feat16/feat32 尺寸相同，这里不再做上采样
#
# 融合逻辑:
# feat16 -> 3x3 conv -> 128
# feat32 -> 3x3 conv -> 128
# concat(feat8, feat16_out, feat32_out) -> 3x3 conv -> 128
# =========================================================
class SPNoDownsample(nn.Module):
    def __init__(self, use_dilation=False, *args, **kwargs):
        super(SPNoDownsample, self).__init__()
        self.resnet = Resnet18NoDownsample(use_dilation=use_dilation)

        # feat16: 256 -> 128
        self.conv16 = ConvBNReLU(256, 128, ks=3, stride=1, padding=1)

        # feat32: 512 -> 128
        self.conv32 = ConvBNReLU(512, 128, ks=3, stride=1, padding=1)

        # feat8(128) + feat16_out(128) + feat32_out(128) = 384
        self.fusion_conv = ConvBNReLU(128 * 3, 128, ks=3, stride=1, padding=1)

        self.init_weight()

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)

        feat16_out = self.conv16(feat16)
        feat32_out = self.conv32(feat32)

        fused_feat = torch.cat([feat8, feat16_out, feat32_out], dim=1)
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


# =========================================================
# test
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("==== test backbone ====")
    backbone = Resnet18NoDownsample(use_dilation=False).to(device)
    backbone.eval()

    x = torch.randn(2, 3, 640, 480).to(device)

    with torch.no_grad():
        feat8, feat16, feat32 = backbone(x)

    print("input  :", x.shape)
    print("feat8  :", feat8.shape)    # [B, 128, H/4, W/4]
    print("feat16 :", feat16.shape)   # [B, 256, H/4, W/4]
    print("feat32 :", feat32.shape)   # [B, 512, H/4, W/4]")

    print("\n==== test CP ====")
    net = CPNoDownsample(use_dilation=False).to(device)
    net.eval()

    with torch.no_grad():
        out = net(x)

    print("cp out :", out.shape)      # [B, 128, H/4, W/4]
    print("forward success.")
