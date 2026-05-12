# Copyright (c) OpenMMLab. All rights reserved.
#
# This file ports the PaddleSeg RTFormer implementation to the MMSegmentation
# segmentor interface used by this repository.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import trunc_normal_init
from mmcv.runner import auto_fp16, load_checkpoint

from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import SEGMENTORS, build_loss
from ..losses import accuracy
from ...utils import get_root_logger
from .base import BaseSegmentor


class DropPath(nn.Module):
    """Drop paths per sample."""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def conv2d(in_channels,
           out_channels,
           kernel_size,
           stride=1,
           padding=0,
           bias=False,
           **kwargs):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        **kwargs)


def bn2d(in_channels, momentum=0.1, eps=1e-5):
    return nn.BatchNorm2d(in_channels, momentum=momentum, eps=eps)


def init_weights_kaiming(module):
    if isinstance(module, nn.Linear):
        trunc_normal_init(module.weight, std=.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class BasicBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv1 = conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = bn2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = bn2d(out_channels)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        return out if self.no_relu else self.relu(out)


class MLP(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = bn2d(in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class ExternalAttention(nn.Module):
    """External attention used by RTFormer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8,
                 use_cross_kv=False):
        super().__init__()
        assert out_channels % num_heads == 0, (
            f'out_channels ({out_channels}) should be a multiple of '
            f'num_heads ({num_heads})')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.use_cross_kv = use_cross_kv
        self.norm = bn2d(in_channels)
        self.same_in_out_chs = in_channels == out_channels

        if use_cross_kv:
            assert self.same_in_out_chs, (
                'in_channels must equal out_channels when use_cross_kv=True')
        else:
            self.k = nn.Parameter(
                torch.empty(inter_channels, in_channels, 1, 1))
            self.v = nn.Parameter(
                torch.empty(out_channels, inter_channels, 1, 1))
            trunc_normal_init(self.k, std=.001)
            trunc_normal_init(self.v, std=.001)

    def _act_sn(self, x):
        n, c, h, w = x.shape
        x = x.reshape(-1, self.inter_channels, h, w)
        x = x * (self.inter_channels**-0.5)
        x = F.softmax(x, dim=1)
        x = x.reshape(1, n * c, h, w)
        return x

    def _act_dn(self, x):
        n, _, h, w = x.shape
        x = x.reshape(n, self.num_heads,
                      self.inter_channels // self.num_heads, -1)
        x = F.softmax(x, dim=3)
        x = x / (x.sum(dim=2, keepdim=True) + 1e-6)
        x = x.reshape(n, self.inter_channels, h, w)
        return x

    def forward(self, x, cross_k=None, cross_v=None):
        x = self.norm(x)
        if not self.use_cross_kv:
            stride = 1 if self.same_in_out_chs else 2
            x = F.conv2d(x, self.k, bias=None, stride=stride, padding=0)
            x = self._act_dn(x)
            x = F.conv2d(x, self.v, bias=None, stride=1, padding=0)
        else:
            assert cross_k is not None and cross_v is not None, (
                'cross_k and cross_v must be set when use_cross_kv=True')
            batch_size, _, h, w = x.shape
            assert batch_size > 0, 'batch size must be greater than 0'
            x = x.reshape(1, -1, h, w)
            x = F.conv2d(
                x, cross_k, bias=None, stride=1, padding=0, groups=batch_size)
            x = self._act_sn(x)
            x = F.conv2d(
                x, cross_v, bias=None, stride=1, padding=0,
                groups=batch_size)
            x = x.reshape(batch_size, self.in_channels, h, w)
        return x


class EABlock(nn.Module):
    """RTFormer external attention block."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_injection=True,
                 use_cross_kv=True,
                 cross_size=12):
        super().__init__()
        in_channels_h, in_channels_l = in_channels
        out_channels_h, out_channels_l = out_channels
        assert in_channels_h == out_channels_h, (
            'in_channels_h must equal out_channels_h')

        self.out_channels_h = out_channels_h
        self.proj_flag = in_channels_l != out_channels_l
        self.use_injection = use_injection
        self.use_cross_kv = use_cross_kv
        self.cross_size = cross_size

        if self.proj_flag:
            self.attn_shortcut_l = nn.Sequential(
                bn2d(in_channels_l),
                conv2d(in_channels_l, out_channels_l, 1, 2, 0))

        self.attn_l = ExternalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=out_channels_l,
            num_heads=num_heads,
            use_cross_kv=False)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate)

        self.compression = nn.Sequential(
            bn2d(out_channels_l),
            nn.ReLU(inplace=False),
            conv2d(out_channels_l, out_channels_h, kernel_size=1))

        self.attn_h = ExternalAttention(
            in_channels_h,
            in_channels_h,
            inter_channels=cross_size * cross_size,
            num_heads=num_heads,
            use_cross_kv=use_cross_kv)
        self.mlp_h = MLP(out_channels_h, drop_rate=drop_rate)

        if use_cross_kv:
            self.cross_kv = nn.Sequential(
                bn2d(out_channels_l),
                nn.AdaptiveMaxPool2d((self.cross_size, self.cross_size)),
                conv2d(out_channels_l, 2 * out_channels_h, 1, 1, 0))

        if use_injection:
            self.down = nn.Sequential(
                bn2d(out_channels_h),
                nn.ReLU(inplace=False),
                conv2d(
                    out_channels_h,
                    out_channels_l // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1),
                bn2d(out_channels_l // 2),
                nn.ReLU(inplace=False),
                conv2d(
                    out_channels_l // 2,
                    out_channels_l,
                    kernel_size=3,
                    stride=2,
                    padding=1))

    def forward(self, x):
        x_h, x_l = x

        x_l_res = self.attn_shortcut_l(x_l) if self.proj_flag else x_l
        x_l = x_l_res + self.drop_path(self.attn_l(x_l))
        x_l = x_l + self.drop_path(self.mlp_l(x_l))

        x_l_cp = self.compression(x_l)
        x_h = x_h + resize(
            x_l_cp,
            size=x_h.shape[2:],
            mode='bilinear',
            align_corners=False)

        if not self.use_cross_kv:
            x_h = x_h + self.drop_path(self.attn_h(x_h))
        else:
            cross_kv = self.cross_kv(x_l)
            cross_k, cross_v = torch.chunk(cross_kv, 2, dim=1)
            cross_k = cross_k.permute(0, 2, 3, 1).reshape(
                -1, self.out_channels_h, 1, 1)
            cross_v = cross_v.reshape(
                -1, self.cross_size * self.cross_size, 1, 1)
            x_h = x_h + self.drop_path(self.attn_h(x_h, cross_k, cross_v))

        x_h = x_h + self.drop_path(self.mlp_h(x_h))

        if self.use_injection:
            x_l = x_l + self.down(x_h)

        return x_h, x_l


class DAPPM(nn.Module):

    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2,
                         count_include_pad=True),
            bn2d(in_channels),
            nn.ReLU(inplace=False),
            conv2d(in_channels, inter_channels, kernel_size=1))
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4,
                         count_include_pad=True),
            bn2d(in_channels),
            nn.ReLU(inplace=False),
            conv2d(in_channels, inter_channels, kernel_size=1))
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8,
                         count_include_pad=True),
            bn2d(in_channels),
            nn.ReLU(inplace=False),
            conv2d(in_channels, inter_channels, kernel_size=1))
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            bn2d(in_channels),
            nn.ReLU(inplace=False),
            conv2d(in_channels, inter_channels, kernel_size=1))
        self.scale0 = nn.Sequential(
            bn2d(in_channels),
            nn.ReLU(inplace=False),
            conv2d(in_channels, inter_channels, kernel_size=1))
        self.process1 = nn.Sequential(
            bn2d(inter_channels),
            nn.ReLU(inplace=False),
            conv2d(inter_channels, inter_channels, kernel_size=3, padding=1))
        self.process2 = nn.Sequential(
            bn2d(inter_channels),
            nn.ReLU(inplace=False),
            conv2d(inter_channels, inter_channels, kernel_size=3, padding=1))
        self.process3 = nn.Sequential(
            bn2d(inter_channels),
            nn.ReLU(inplace=False),
            conv2d(inter_channels, inter_channels, kernel_size=3, padding=1))
        self.process4 = nn.Sequential(
            bn2d(inter_channels),
            nn.ReLU(inplace=False),
            conv2d(inter_channels, inter_channels, kernel_size=3, padding=1))
        self.compression = nn.Sequential(
            bn2d(inter_channels * 5),
            nn.ReLU(inplace=False),
            conv2d(inter_channels * 5, out_channels, kernel_size=1))
        self.shortcut = nn.Sequential(
            bn2d(in_channels),
            nn.ReLU(inplace=False),
            conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x_shape = x.shape[2:]
        x_list = [self.scale0(x)]
        x_list.append(self.process1(
            resize(
                self.scale1(x),
                size=x_shape,
                mode='bilinear',
                align_corners=False) + x_list[0]))
        x_list.append(self.process2(
            resize(
                self.scale2(x),
                size=x_shape,
                mode='bilinear',
                align_corners=False) + x_list[1]))
        x_list.append(self.process3(
            resize(
                self.scale3(x),
                size=x_shape,
                mode='bilinear',
                align_corners=False) + x_list[2]))
        x_list.append(self.process4(
            resize(
                self.scale4(x),
                size=x_shape,
                mode='bilinear',
                align_corners=False) + x_list[3]))

        out = self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)
        return out


class SegHead(nn.Module):

    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.bn1 = bn2d(in_channels)
        self.conv1 = conv2d(
            in_channels, inter_channels, kernel_size=3, padding=1)
        self.bn2 = bn2d(inter_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv2d(
            inter_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=True)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return out


@SEGMENTORS.register_module()
class RTFormer(BaseSegmentor):
    """RTFormer segmentor with built-in heads for MMSegmentation.

    The implementation follows RTFormer: Efficient Design for Real-Time
    Semantic Segmentation with Transformer.
    """

    def __init__(self,
                 num_classes,
                 layer_nums=(2, 2, 2, 2),
                 base_channels=64,
                 spp_channels=128,
                 num_heads=8,
                 head_channels=128,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 use_aux_head=True,
                 use_injection=(True, True),
                 cross_size=12,
                 in_channels=3,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_aux=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=0.4),
                 ignore_index=255,
                 align_corners=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        assert len(layer_nums) == 4
        assert len(use_injection) == 2

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.ignore_index = ignore_index
        self.use_aux_head = use_aux_head
        self.pretrained = pretrained
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        base_chs = base_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, base_chs, kernel_size=3, stride=2, padding=1),
            bn2d(base_chs),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                base_chs, base_chs, kernel_size=3, stride=2, padding=1),
            bn2d(base_chs),
            nn.ReLU(inplace=False))
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(BasicBlock, base_chs, base_chs,
                                       layer_nums[0])
        self.layer2 = self._make_layer(
            BasicBlock, base_chs, base_chs * 2, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(
            BasicBlock, base_chs * 2, base_chs * 4, layer_nums[2], stride=2)
        self.layer3_ = self._make_layer(
            BasicBlock, base_chs * 2, base_chs * 2, 1)
        self.compression3 = nn.Sequential(
            bn2d(base_chs * 4),
            nn.ReLU(inplace=False),
            conv2d(base_chs * 4, base_chs * 2, kernel_size=1))
        self.layer4 = EABlock(
            in_channels=[base_chs * 2, base_chs * 4],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[0],
            use_cross_kv=True,
            cross_size=cross_size)
        self.layer5 = EABlock(
            in_channels=[base_chs * 2, base_chs * 8],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[1],
            use_cross_kv=True,
            cross_size=cross_size)

        self.spp = DAPPM(base_chs * 8, spp_channels, base_chs * 2)
        self.seghead = SegHead(base_chs * 4, int(head_channels * 2),
                               num_classes)
        if self.use_aux_head:
            self.seghead_extra = SegHead(base_chs * 2, head_channels,
                                         num_classes)

        self.loss_decode = build_loss(loss_decode)
        self.loss_aux = build_loss(loss_aux) if loss_aux is not None else None

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                conv2d(in_channels, out_channels, kernel_size=1,
                       stride=stride),
                bn2d(out_channels))

        layers = [block(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            no_relu = i == (blocks - 1)
            layers.append(
                block(out_channels, out_channels, stride=1,
                      no_relu=no_relu))
        return nn.Sequential(*layers)

    def init_weights(self):
        self.apply(init_weights_kaiming)
        for module in self.modules():
            if isinstance(module, ExternalAttention) and not module.use_cross_kv:
                trunc_normal_init(module.k, std=.001)
                trunc_normal_init(module.v, std=.001)

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    def extract_feat(self, img):
        return self._forward_logits(img, return_aux=False)[0]

    def _forward_logits(self, x, return_aux=False):
        x1 = self.layer1(self.conv1(x))
        x2 = self.layer2(self.relu(x1))
        x3 = self.layer3(self.relu(x2))
        x3_ = x2 + resize(
            self.compression3(x3),
            size=x2.shape[2:],
            mode='bilinear',
            align_corners=False)
        x3_ = self.layer3_(self.relu(x3_))

        x4_, x4 = self.layer4([self.relu(x3_), self.relu(x3)])
        x5_, x5 = self.layer5([self.relu(x4_), self.relu(x4)])

        x6 = self.spp(x5)
        x6 = resize(
            x6, size=x5_.shape[2:], mode='bilinear', align_corners=False)
        x_out = self.seghead(torch.cat([x5_, x6], dim=1))
        logits = [x_out]

        if return_aux and self.use_aux_head:
            logits.append(self.seghead_extra(x3_))

        logits = [
            resize(
                logit,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logits
        ]
        return logits

    def _loss_by_feat(self, seg_logit, seg_label, loss_module):
        seg_logit = resize(
            seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)
        loss = dict()
        loss['loss_seg'] = loss_module(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

    @auto_fp16(apply_to=('img', ))
    def forward_train(self, img, img_metas, gt_semantic_seg):
        logits = self._forward_logits(img, return_aux=True)
        losses = dict()
        losses.update(
            add_prefix(
                self._loss_by_feat(logits[0], gt_semantic_seg,
                                   self.loss_decode), 'decode'))

        if self.use_aux_head and self.loss_aux is not None and len(logits) > 1:
            losses.update(
                add_prefix(
                    self._loss_by_feat(logits[1], gt_semantic_seg,
                                       self.loss_aux), 'aux'))
        return losses

    def encode_decode(self, img, img_metas):
        return self._forward_logits(img, return_aux=False)[0]

    def forward_dummy(self, img):
        return self.encode_decode(img, None)

    def slide_inference(self, img, img_meta, rescale):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        preds = img.new_zeros((batch_size, self.num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            size = img.shape[2:] if torch.onnx.is_in_onnx_export() else \
                img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return seg_logit

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            else:
                output = output.flip(dims=(2, ))
        return output

    def simple_test(self, img, img_meta, rescale=True):
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            return seg_pred.unsqueeze(0)
        seg_pred = seg_pred.cpu().numpy()
        return list(seg_pred)

    def aug_test(self, imgs, img_metas, rescale=True):
        assert rescale
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            seg_logit = seg_logit + self.inference(imgs[i], img_metas[i],
                                                   rescale)
        seg_logit = seg_logit / len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        return list(seg_pred)
