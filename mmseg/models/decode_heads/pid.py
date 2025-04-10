# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.ops import resize
from .harr import DWT,IWT
from mmcv.cnn import ConvModule
from mmcv.ops.carafe import normal_init, xavier_init, carafe

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    expansion = 1
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):

        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out

class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*4, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes*4, branch_planes*4, kernel_size=3, padding=1, groups=4, bias=False),
                                    )


        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )

        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )


    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)

        scale_out = self.scale_process(torch.cat(scale_list, 1))

        out = self.compression(torch.cat([x_,scale_out], 1)) + self.shortcut(x)
        return out

# class harr_fusion(nn.Module):
#     def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels,
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_y = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels,
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        if with_channel:
            self.up = nn.Sequential(
                                    nn.Conv2d(mid_channels, in_channels,
                                              kernel_size=1, bias=False),
                                    BatchNorm(in_channels)
                                   )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x = (1-sim_map)*x + sim_map*y

        return x



class UncertaintyFusion(nn.Module):
    def __init__(self, in_channels,  after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(UncertaintyFusion, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.bn = BatchNorm2d(in_channels, momentum=bn_mom)
        self.add=AddFuse(in_channels,in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        # y_ori=y
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        # dot_product = torch.sum(x * y, dim=1, keepdim=True)
        # norm_x = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        # norm_y = torch.sqrt(torch.sum(y ** 2, dim=1, keepdim=True))
        # cosine_similarity = dot_product / (norm_x * norm_y + 1e-8)
        # uncertainty_mask=torch.sigmoid(cosine_similarity)
        # uncertainty_mask=1-uncertainty_mask
        x=self.add(x,y)
        x=self.bn(x)
        # x=x+x*uncertainty_mask
        # uncertainty_mask = F.interpolate(uncertainty_mask, size=[y_ori.size()[2], y_ori.size()[3]],
        #                     mode='bilinear', align_corners=False)
        return x

class Light_Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Light_Bag, self).__init__()
        self.conv_p = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        self.conv_i = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)

        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(i + edge_att*p)

        return p_add + i_add


class DDFMv2(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(DDFMv2, self).__init__()
        self.conv_p = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )
        self.conv_i = nn.Sequential(
                                BatchNorm(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, bias=False),
                                BatchNorm(out_channels)
                                )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)

        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(i + edge_att*p)

        return p_add + i_add


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Bag, self).__init__()

        self.conv = nn.Sequential(
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)



class HarrFusion(nn.Module):
    def __init__(self, p_channels,i_channels, BatchNorm=nn.BatchNorm2d):
        super(HarrFusion, self).__init__()
        self.dwt=DWT().cuda()
        self.iwt=IWT().cuda()
        self.conv_p = nn.Sequential(
            BatchNorm(p_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(p_channels*4, p_channels*4,
                      kernel_size=3, padding=1, bias=False)
        )
        self.conv_i = nn.Sequential(
            BatchNorm(i_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(i_channels*4, i_channels*4,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, p, i):
        p_LL,p_HL,p_LH,p_HH=self.dwt(p)
        p_high=torch.concat((p_HL,p_LH,p_HH),dim=1)
        i_LL,i_HL,i_LH,i_HH=self.dwt(i)
        p_LL=F.interpolate(i_LL, size=p_LL.size()[2:],mode='bilinear', align_corners=False)
        p_harr=torch.concat((p_LL,p_high),dim=1)
        i_high=F.interpolate(p_high, size=i_LL.size()[2:],mode='bilinear', align_corners=False)
        i_harr=torch.concat((i_LL,i_high),dim=1)
        p_harr=self.conv_p(p_harr)
        i_harr=self.conv_i(i_harr)
        p=self.iwt(p_harr)
        i=self.iwt(p_harr)
        return p,i



        return self.conv(edge_att * p + (1 - edge_att) * i)

class AddFuse(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(AddFuse, self).__init__()

        self.conv = nn.Sequential(
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, p, i):

        return self.conv(p+i)

class AddFuse_sample(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(AddFuse_sample, self).__init__()

        self.conv = nn.Sequential(
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x, y):
        input_size = x.size()
        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        return self.conv(x+y)

class ConcatFuse(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(ConcatFuse, self).__init__()

        self.conv = nn.Sequential(
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False)
        )

    def forward(self, p, i):
        cat=torch.concat([p,i],dim=1)
        return self.conv(cat)
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
    def __init__(self, channels, conv_cfg, norm_cfg, act_cfg, ext=1, r=16):
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
        out = self.smooth(s_feat + c_feat)
        return s_feat, c_feat, out

class AdaptiveFrequencyFusion(nn.Module):
    def __init__(self, sp_channels,co_channels,out_channels,mid_channels,kernel_size=3):
        super(AdaptiveFrequencyFusion, self).__init__()
        self.kernel_size=kernel_size
        self.sp_compressor=nn.Conv2d(sp_channels, mid_channels,1)
        self.co_compressor=nn.Conv2d(co_channels, mid_channels,1)
        self.adaptive_sp_conv=nn.Conv2d(mid_channels,kernel_size**2,stride=1,kernel_size=3,padding=kernel_size//2)
        self.adaptive_co_conv=nn.Conv2d(mid_channels,kernel_size**2,stride=1,kernel_size=3,padding=kernel_size//2)

        self.adaptive_sp_co_conv=nn.Conv2d(mid_channels,kernel_size**2,stride=1,kernel_size=3,padding=kernel_size//2)
        self.adaptive_co_sp_conv=nn.Conv2d(mid_channels,kernel_size**2,stride=1,kernel_size=3,padding=kernel_size//2)


        # self.addConv = ConvModule(mid_channels, out_channels, stride=1, kernel_size=3, padding=1)
        self.addConv=AddFuse(mid_channels,out_channels)
        self.cat=ConcatFuse(mid_channels*2,out_channels)

    def hamming2D(M, N):
        """
        生成二维Hamming窗

        参数：
        - M：窗口的行数
        - N：窗口的列数

        返回：
        - 二维Hamming窗
        """
        # 生成水平和垂直方向上的Hamming窗
        # hamming_x = np.blackman(M)
        # hamming_x = np.kaiser(M)
        hamming_x = np.hamming(M)
        hamming_y = np.hamming(N)
        # 通过外积生成二维Hamming窗
        hamming_2d = np.outer(hamming_x, hamming_y)
        return hamming_2d

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))  # group
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        # print(hamming)
        # print(mask.shape)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, sp_feat, co_feat):
        sp_feat_compress=self.sp_compressor(sp_feat)
        co_feat_compress=self.co_compressor(co_feat)
        # co_feat_compress=F.interpolate(co_feat_compress, size=sp_feat_compress.shape[-2:], mode='bilinear')
        # sp_mask=self.adaptive_sp_conv(sp_feat_compress)

        sp_mask=self.adaptive_sp_conv(sp_feat_compress)
        sp_mask=self.kernel_normalizer(sp_mask,self.kernel_size)
        sp_feat=carafe(sp_feat_compress,sp_mask,self.kernel_size,1,1)

        co_mask=self.adaptive_co_conv(co_feat_compress)
        co_mask=self.kernel_normalizer(co_mask,self.kernel_size )
        co_feat=carafe(co_feat_compress, co_mask, self.kernel_size, 1, 1)
        co_feat=F.interpolate(co_feat, size=sp_feat.shape[-2:], mode='bilinear')

        # sp_co_mask=self.adaptive_sp_co_conv(sp_feat)
        # sp_co_mask = self.kernel_normalizer(sp_co_mask, self.kernel_size)
        # co_feat=carafe(co_feat,sp_co_mask,self.kernel_size,1,1)
        #
        # co_sp_mask=self.adaptive_co_sp_conv(co_feat)
        # co_sp_mask=self.kernel_normalizer(co_sp_mask,self.kernel_size)
        # sp_feat=carafe(sp_feat,co_sp_mask,self.kernel_size,1,1)

        fusion=self.addConv(co_feat,sp_feat)
        # fusion=self.cat(sp_feat,co_feat)
        return fusion


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x




if __name__ == '__main__':


    x = torch.rand(4, 64, 32, 64).cuda()
    y = torch.rand(4, 64, 32, 64).cuda()
    z = torch.rand(4, 64, 32, 64).cuda()
    net = PagFM(64, 16, with_channel=True).cuda()

    out = net(x,y)
