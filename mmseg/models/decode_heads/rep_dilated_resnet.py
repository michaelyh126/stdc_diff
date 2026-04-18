import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d


class ConvBN(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.bn = BatchNorm2d(out_chan)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ParallelDilatedBasicBlock(nn.Module):
    """
    并行分支:
        out = branch_3x3(x) + branch_dilated_3x3(x) + shortcut(x)
        out = ReLU(out)

    其中:
        - branch_3x3: 普通 3x3 卷积
        - branch_dilated_3x3: dilation=2 的 3x3 卷积
        - shortcut: 残差分支
    """
    expansion = 1

    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        # 普通 3x3
        self.branch_a = ConvBN(
            in_chan,
            out_chan,
            ks=3,
            stride=stride,
            padding=1,
            dilation=1
        )

        # 膨胀 3x3, dilation=2，对应 padding 要设为 2
        self.branch_b = ConvBN(
            in_chan,
            out_chan,
            ks=3,
            stride=stride,
            padding=2,
            dilation=2
        )

        # shortcut
        if stride == 1 and in_chan == out_chan:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ConvBN(
                in_chan,
                out_chan,
                ks=1,
                stride=stride,
                padding=0,
                dilation=1
            )

    def forward(self, x):
        out = self.branch_a(x) + self.branch_b(x)

        if isinstance(self.shortcut, nn.Identity):
            out = out + x
        else:
            out = out + self.shortcut(x)

        out = self.relu(out)
        return out


class ParallelDilatedResNet18(nn.Module):
    """
    结构上参考你给的 RepResnet18，
    但 block 改为:
        普通3x3 + dilation=2的3x3 + residual

    输出:
        feat8  : 1/8,  128 channels
        feat16 : 1/16, 256 channels
        feat32 : 1/32, 512 channels
    """
    def __init__(self):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.layer1 = self._make_layer(64,  blocks=2, stride=1)  # 1/4
        self.layer2 = self._make_layer(128, blocks=2, stride=2)  # 1/8
        self.layer3 = self._make_layer(256, blocks=2, stride=2)  # 1/16
        self.layer4 = self._make_layer(512, blocks=2, stride=2)  # 1/32

        self.init_weight()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [ParallelDilatedBasicBlock(self.inplanes, planes, stride=stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ParallelDilatedBasicBlock(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 1/4

        x = self.layer1(x)         # 1/4
        feat8 = self.layer2(x)     # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16) # 1/32

        return feat8, feat16, feat32

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ParallelDilatedSP(nn.Module):
    """
    和你原来的 RepSP 接口保持一致
    融合方式:
        feat16 -> conv -> up
        feat32 -> conv -> up
        concat(feat8, feat16_up, feat32_up) -> conv
    输出:
        [B, 128, H/8, W/8]
    """
    def __init__(self):
        super().__init__()
        self.resnet = ParallelDilatedResNet18()

        self.conv16 = ConvBNReLU(256, 128, ks=3, stride=1, padding=1)
        self.conv32 = ConvBNReLU(512, 128, ks=3, stride=1, padding=1)
        self.fusion_conv = ConvBNReLU(128 * 3, 128, ks=3, stride=1, padding=1)

        self.init_weight()

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        h8, w8 = feat8.shape[2:]

        feat16_out = self.conv16(feat16)
        feat16_up = F.interpolate(feat16_out, size=(h8, w8), mode='nearest')

        feat32_out = self.conv32(feat32)
        feat32_up = F.interpolate(feat32_out, size=(h8, w8), mode='nearest')

        fused = torch.cat([feat8, feat16_up, feat32_up], dim=1)
        fused = self.fusion_conv(fused)
        return fused

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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


if __name__ == '__main__':
    model = ParallelDilatedSP().eval()
    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        out = model(x)
        print('out shape:', out.shape)

        feat8, feat16, feat32 = model.resnet(x)
        print('feat8 shape :', feat8.shape)
        print('feat16 shape:', feat16.shape)
        print('feat32 shape:', feat32.shape)
