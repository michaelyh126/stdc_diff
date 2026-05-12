import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=5, stride=1, sync=False):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            bias=False)
        if sync:
            self.bn = nn.SyncBatchNorm(out_planes)
        else:
            self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RepStem5x5(nn.Module):
    """Train with three receptive fields, deploy as one 5x5 conv."""

    def __init__(self, in_channels, out_channels, stride=1, sync=False, deploy=False):
        super(RepStem5x5, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.deploy = deploy

        if deploy:
            self.reparam_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=True)
        else:
            self.branch_3x3 = self._conv_bn(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                dilation=1,
                sync=sync)
            self.branch_dilated_3x3 = self._conv_bn(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=2,
                dilation=2,
                sync=sync)
            self.branch_5x5 = self._conv_bn(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                dilation=1,
                sync=sync)

        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _conv_bn(in_channels, out_channels, kernel_size, stride, padding, dilation, sync=False):
        if sync:
            norm = nn.SyncBatchNorm(out_channels)
        else:
            norm = nn.BatchNorm2d(out_channels)
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False),
            norm)

    def forward(self, x):
        if self.deploy:
            return self.relu(self.reparam_conv(x))

        out = self.branch_3x3(x)
        out = out + self.branch_dilated_3x3(x)
        out = out + self.branch_5x5(x)
        return self.relu(out)

    @staticmethod
    def _pad_3x3_to_5x5(kernel):
        return F.pad(kernel, [1, 1, 1, 1])

    @staticmethod
    def _inflate_dilated_3x3_to_5x5(kernel):
        out_channels, in_channels, _, _ = kernel.shape
        kernel_5x5 = kernel.new_zeros((out_channels, in_channels, 5, 5))
        kernel_5x5[:, :, 0, 0] = kernel[:, :, 0, 0]
        kernel_5x5[:, :, 0, 2] = kernel[:, :, 0, 1]
        kernel_5x5[:, :, 0, 4] = kernel[:, :, 0, 2]
        kernel_5x5[:, :, 2, 0] = kernel[:, :, 1, 0]
        kernel_5x5[:, :, 2, 2] = kernel[:, :, 1, 1]
        kernel_5x5[:, :, 2, 4] = kernel[:, :, 1, 2]
        kernel_5x5[:, :, 4, 0] = kernel[:, :, 2, 0]
        kernel_5x5[:, :, 4, 2] = kernel[:, :, 2, 1]
        kernel_5x5[:, :, 4, 4] = kernel[:, :, 2, 2]
        return kernel_5x5

    def _fuse_conv_bn(self, branch):
        conv = branch[0]
        bn = branch[1]
        kernel = conv.weight

        if conv.kernel_size == (3, 3) and conv.dilation == (1, 1):
            kernel = self._pad_3x3_to_5x5(kernel)
        elif conv.kernel_size == (3, 3) and conv.dilation == (2, 2):
            kernel = self._inflate_dilated_3x3_to_5x5(kernel)
        elif conv.kernel_size == (5, 5) and conv.dilation == (1, 1):
            pass
        else:
            raise ValueError(
                f'Unsupported branch: kernel={conv.kernel_size}, dilation={conv.dilation}')

        if conv.bias is None:
            conv_bias = torch.zeros(kernel.size(0), device=kernel.device, dtype=kernel.dtype)
        else:
            conv_bias = conv.bias

        std = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / std
        fused_kernel = kernel * scale.reshape(-1, 1, 1, 1)
        fused_bias = bn.bias + (conv_bias - bn.running_mean) * scale
        return fused_kernel, fused_bias

    def get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_conv_bn(self.branch_3x3)
        kd, bd = self._fuse_conv_bn(self.branch_dilated_3x3)
        k5, b5 = self._fuse_conv_bn(self.branch_5x5)
        return k3 + kd + k5, b3 + bd + b5

    def switch_to_deploy(self):
        if self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=5,
            stride=self.stride,
            padding=2,
            bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        del self.branch_3x3
        del self.branch_dilated_3x3
        del self.branch_5x5
        self.deploy = True


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print('block number should be larger than 1.')
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_planes,
                    bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)),
                          out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)),
                          out_planes // int(math.pow(2, idx))))

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


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print('block number should be larger than 1.')
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)),
                          out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)),
                          out_planes // int(math.pow(2, idx))))

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

        return torch.cat(out_list, dim=1)


class ShallowNetRepStem(nn.Module):
    def __init__(
            self,
            base=64,
            in_channels=3,
            layers=(2, 2),
            block_num=4,
            type='cat',
            dropout=0.20,
            num_classes=2,
            deploy=False):
        super(ShallowNetRepStem, self).__init__()
        if type == 'cat':
            block = CatBottleneck
        elif type == 'add':
            block = AddBottleneck
        else:
            raise ValueError(f'Unsupported STDC block type: {type}')

        self.in_channels = in_channels
        self.deploy = deploy
        self.cls_feat16 = ConvX(512, num_classes, 3, 1)

        self.features = self._make_layers(base, layers, block_num, block, deploy=deploy)
        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])

    def _make_layers(self, base, layers, block_num, block, deploy=False):
        features = [
            RepStem5x5(self.in_channels, base // 2, stride=2, deploy=deploy),
            RepStem5x5(base // 2, base, stride=2, deploy=deploy),
        ]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(
                            base * int(math.pow(2, i + 1)),
                            base * int(math.pow(2, i + 2)),
                            block_num,
                            2))
                else:
                    features.append(
                        block(
                            base * int(math.pow(2, i + 2)),
                            base * int(math.pow(2, i + 2)),
                            block_num,
                            1))

        return nn.Sequential(*features)

    def switch_to_deploy(self):
        if self.deploy:
            return

        memo = set()
        for module in self.modules():
            if module is self or module in memo:
                continue
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
                memo.add(module)
        self.deploy = True

    def forward(self, x, cas3=False):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        if cas3:
            return feat4, feat8, feat16
        return feat8, feat16


ShallowNetStemRep = ShallowNetRepStem
