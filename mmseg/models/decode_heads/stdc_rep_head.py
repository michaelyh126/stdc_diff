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

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class RepSuperBranch(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_branches=3, deploy=False):
        super(RepSuperBranch, self).__init__()
        self.deploy = deploy
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_branches = num_branches
        self.padding = 1

        if deploy:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, 3, stride, self.padding, bias=True)
        else:
            # 训练分支：多组 (3x3 + 1x1)
            self.branches_3x3 = nn.ModuleList([
                self._conv_bn(in_channels, out_channels, 3, stride, 1) for _ in range(num_branches)
            ])
            self.branches_1x1 = nn.ModuleList([
                self._conv_bn(in_channels, out_channels, 1, stride, 0) for _ in range(num_branches)
            ])
            # 残差分支
            self.identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

        self.relu = nn.ReLU(inplace=True)

    def _conv_bn(self, in_c, out_c, k, s, p):
        module = nn.Sequential()
        module.add_module('conv', nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False))
        module.add_module('bn', nn.BatchNorm2d(out_c))
        return module

    def forward(self, x):
        if self.deploy:
            return self.relu(self.reparam_conv(x))

        out = 0
        for b3, b1 in zip(self.branches_3x3, self.branches_1x1):
            out += (b3(x) + b1(x))

        if self.identity is not None:
            out += self.identity(x)

        return self.relu(out)

    def _fuse_bn_tensor(self, branch):
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            bn = branch.bn
            if kernel.shape[2] == 1:
                kernel = F.pad(kernel, [1, 1, 1, 1])
        elif isinstance(branch, nn.BatchNorm2d):
            input_dim = self.in_channels
            kernel = torch.zeros((input_dim, input_dim, 3, 3), device=branch.weight.device)
            for i in range(input_dim):
                kernel[i, i, 1, 1] = 1
            bn = branch
        else:
            return 0, 0

        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def switch_to_deploy(self):
        if self.deploy: return
        combined_kernel, combined_bias = 0, 0
        for b3, b1 in zip(self.branches_3x3, self.branches_1x1):
            k3, b3_v = self._fuse_bn_tensor(b3)
            k1, b1_v = self._fuse_bn_tensor(b1)
            combined_kernel += (k3 + k1)
            combined_bias += (b3_v + b1_v)

        if self.identity is not None:
            kid, bid = self._fuse_bn_tensor(self.identity)
            combined_kernel += kid
            combined_bias += bid

        self.reparam_conv = nn.Conv2d(self.in_channels, self.out_channels, 3, self.stride, self.padding, bias=True)
        self.reparam_conv.weight.data = combined_kernel
        self.reparam_conv.bias.data = combined_bias

        # 清理训练分支
        del self.branches_3x3
        del self.branches_1x1
        if hasattr(self, 'identity'): del self.identity
        self.deploy = True


class CatBottleneck_Rep(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=4, stride=1, num_branches=4, deploy=False):
        super(CatBottleneck_Rep, self).__init__()
        assert block_num > 1
        self.conv_list = nn.ModuleList()
        self.stride = stride
        self.out_planes = out_planes

        # 下采样层
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, 3, 2, 1, groups=out_planes // 2, bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        # 内部重参数化路径
        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            else:
                cur_in = out_planes // 2 if idx == 1 else out_planes // (2 ** idx)
                cur_out = out_planes // (2 ** (idx + 1)) if idx < block_num - 1 else out_planes // (2 ** idx)
                self.conv_list.append(
                    RepSuperBranch(cur_in, cur_out, stride=stride, num_branches=num_branches, deploy=deploy))

        # 宏观残差对齐 (Shortcut)
        self.downsample = None
        if self.stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                out = conv(self.avd_layer(out1)) if self.stride == 2 else conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)

        out_list.insert(0, out1)
        out = torch.cat(out_list, dim=1)

        # 应用宏观残差
        if self.downsample is not None:
            identity = self.downsample(x)

        return self.final_relu(out + identity)

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepSuperBranch):
                m.switch_to_deploy()


class ShallowNet_rep(nn.Module):
    def __init__(self, base=64, in_channels=3,  layers=[2,2], block_num=4, dropout=0.20, pretrain_model="D:\isd\pretrained models\STDCNet813M_73.91.tar",num_classes=2):
        super(ShallowNet_rep, self).__init__()
        block = CatBottleneck_Rep
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

    def switch_to_deploy(self):
        # 1. 检查是否已经转换过，防止重复执行
        if hasattr(self, 'deploy') and self.deploy:
            return
        # 2. 使用 set 来存储已经处理过的模块，防止因为 x8, x16 引用了 features 里的相同层而重复调用
        memo = set()

        for m in self.modules():
            # 关键点 1: m is not self -> 排除掉模型自身，防止递归死循环
            # 关键点 2: m not in memo -> 确保同一个层只转换一次
            if m is not self and m not in memo:
                if hasattr(m, 'switch_to_deploy'):
                    m.switch_to_deploy()
                    memo.add(m)

        self.deploy = True

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

