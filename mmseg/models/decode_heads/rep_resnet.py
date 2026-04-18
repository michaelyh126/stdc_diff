import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d


class ConvBN(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan,
            kernel_size=ks, stride=stride, padding=padding,
            groups=groups, bias=False
        )
        self.bn = BatchNorm2d(out_chan)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan,
            kernel_size=ks, stride=stride, padding=padding,
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




class RepParallelBasicBlock(nn.Module):
    """
    训练时:
        out = branch3x3_a(x) + branch3x3_b(x) + shortcut(x)
        out = ReLU(out)

    部署时:
        等价融合成一个 3x3 Conv(bias=True) + ReLU
    """
    expansion = 1

    def __init__(self, in_chan, out_chan, stride=1, deploy=False):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.stride = stride
        self.deploy = deploy
        self.relu = nn.ReLU(inplace=True)

        if deploy:
            self.reparam_conv = nn.Conv2d(
                in_chan, out_chan,
                kernel_size=3, stride=stride, padding=1,
                bias=True
            )
        else:
            self.branch_a = ConvBN(in_chan, out_chan, ks=3, stride=stride, padding=1)
            self.branch_b = ConvBN(in_chan, out_chan, ks=3, stride=stride, padding=1)

            if stride == 1 and in_chan == out_chan:
                # identity + BN，便于融合
                self.shortcut = BatchNorm2d(in_chan)
            else:
                self.shortcut = ConvBN(in_chan, out_chan, ks=1, stride=stride, padding=0)

    def forward(self, x):
        if self.deploy:
            return self.relu(self.reparam_conv(x))

        out = self.branch_a(x) + self.branch_b(x) + self.shortcut(x)
        return self.relu(out)

    # -------------------------
    # re-parameterization
    # -------------------------
    def _fuse_conv_bn(self, conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)

        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std
        return fused_kernel, fused_bias

    def _fuse_identity_bn(self, bn):
        assert self.in_chan == self.out_chan
        kernel = torch.zeros(
            (self.out_chan, self.in_chan, 3, 3),
            dtype=bn.weight.dtype,
            device=bn.weight.device
        )
        for i in range(self.out_chan):
            kernel[i, i, 1, 1] = 1.0

        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)

        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std
        return fused_kernel, fused_bias

    def _pad_1x1_to_3x3(self, kernel):
        return F.pad(kernel, [1, 1, 1, 1])

    def _get_equivalent_kernel_bias_of_branch(self, branch):
        if isinstance(branch, ConvBN):
            k, b = self._fuse_conv_bn(branch.conv, branch.bn)
            if k.size(2) == 1:
                k = self._pad_1x1_to_3x3(k)
            return k, b

        if isinstance(branch, BatchNorm2d):
            return self._fuse_identity_bn(branch)

        raise TypeError(f"Unsupported branch type: {type(branch)}")

    def get_equivalent_kernel_bias(self):
        k1, b1 = self._get_equivalent_kernel_bias_of_branch(self.branch_a)
        k2, b2 = self._get_equivalent_kernel_bias_of_branch(self.branch_b)
        ks, bs = self._get_equivalent_kernel_bias_of_branch(self.shortcut)
        return k1 + k2 + ks, b1 + b2 + bs

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()

        self.reparam_conv = nn.Conv2d(
            self.in_chan, self.out_chan,
            kernel_size=3, stride=self.stride, padding=1,
            bias=True
        )
        self.reparam_conv.weight.data.copy_(kernel)
        self.reparam_conv.bias.data.copy_(bias)

        del self.branch_a
        del self.branch_b
        del self.shortcut

        self.deploy = True


class RepResnet18(nn.Module):
    """
    输出:
        feat8  : 1/8,  128 channels
        feat16 : 1/16, 256 channels
        feat32 : 1/32, 512 channels
    和你上传代码里的 Resnet18 输出接口保持一致。:contentReference[oaicite:1]{index=1}
    """
    def __init__(self, deploy=False):
        super().__init__()
        self.inplanes = 64
        self.deploy = deploy

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  2, stride=1, deploy=deploy)  # 1/4
        self.layer2 = self._make_layer(128, 2, stride=2, deploy=deploy)  # 1/8
        self.layer3 = self._make_layer(256, 2, stride=2, deploy=deploy)  # 1/16
        self.layer4 = self._make_layer(512, 2, stride=2, deploy=deploy)  # 1/32

        self.init_weight()

    def _make_layer(self, planes, blocks, stride=1, deploy=False):
        layers = [RepParallelBasicBlock(self.inplanes, planes, stride=stride, deploy=deploy)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(RepParallelBasicBlock(self.inplanes, planes, stride=1, deploy=deploy))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 1/4

        x = self.layer1(x)       # 1/4
        feat8 = self.layer2(x)   # 1/8
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

    @torch.no_grad()
    def switch_to_deploy(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, RepParallelBasicBlock):
                m.switch_to_deploy()


class RepSP(nn.Module):
    """
    内部直接包含 RepResnet18
    融合方式:
        feat16 -> conv -> up
        feat32 -> conv -> up
        concat(feat8, feat16_up, feat32_up) -> conv
    输出:
        [B, 128, H/8, W/8]
    """
    def __init__(self, deploy_backbone=False):
        super().__init__()
        self.resnet = RepResnet18(deploy=deploy_backbone)

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

    @torch.no_grad()
    def switch_to_deploy(self):
        self.resnet.switch_to_deploy()

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
    model = RepSP().eval()
    x = torch.randn(1, 3, 256, 256)
    model.eval()

    with torch.no_grad():
        out2 = model(x)
        print(out2)
