import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, trunc_normal_init
from mmcv.runner import auto_fp16

from mmseg.core import add_prefix
from mmseg.ops import resize
from ..builder import SEGMENTORS, build_backbone, build_loss
from ..decode_heads.stdc_lk_head import ShallowNet_lk
from ..losses import accuracy
from ...utils import get_root_logger
from .base import BaseSegmentor


def conv2d(in_channels,
           out_channels,
           kernel_size,
           stride=1,
           padding=0,
           bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias)


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
        nn.init.kaiming_normal_(
            module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


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
    """External attention used for cross-resolution interaction."""

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


class CrossResolutionAttention(nn.Module):
    """RTFormer-style cross-resolution attention for a 2x scale pair.

    The high branch is the STDC detail feature, and the low branch is the
    ResNet semantic feature. Low features generate cross K/V for high features,
    while downsampled high features are injected back into the low branch.
    """

    def __init__(self,
                 high_channels,
                 low_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_injection=True,
                 cross_size=12):
        super().__init__()
        self.high_channels = high_channels
        self.low_channels = low_channels
        self.use_injection = use_injection
        self.cross_size = cross_size

        self.attn_l = ExternalAttention(
            low_channels,
            low_channels,
            inter_channels=low_channels,
            num_heads=num_heads,
            use_cross_kv=False)
        self.mlp_l = MLP(low_channels, drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate)

        self.low_to_high = nn.Sequential(
            bn2d(low_channels),
            nn.ReLU(inplace=False),
            conv2d(low_channels, high_channels, kernel_size=1))
        self.attn_h = ExternalAttention(
            high_channels,
            high_channels,
            inter_channels=cross_size * cross_size,
            num_heads=num_heads,
            use_cross_kv=True)
        self.mlp_h = MLP(high_channels, drop_rate=drop_rate)
        self.cross_kv = nn.Sequential(
            bn2d(low_channels),
            nn.AdaptiveMaxPool2d((self.cross_size, self.cross_size)),
            conv2d(low_channels, 2 * high_channels, 1, 1, 0))

        if use_injection:
            mid_channels = max(low_channels // 2, 1)
            self.high_to_low = nn.Sequential(
                bn2d(high_channels),
                nn.ReLU(inplace=False),
                conv2d(
                    high_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1),
                bn2d(mid_channels),
                nn.ReLU(inplace=False),
                conv2d(
                    mid_channels,
                    low_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))

    def forward(self, high, low):
        low = low + self.drop_path(self.attn_l(low))
        low = low + self.drop_path(self.mlp_l(low))

        high = high + resize(
            self.low_to_high(low),
            size=high.shape[2:],
            mode='bilinear',
            align_corners=False)

        cross_kv = self.cross_kv(low)
        cross_k, cross_v = torch.chunk(cross_kv, 2, dim=1)
        cross_k = cross_k.permute(0, 2, 3, 1).reshape(
            -1, self.high_channels, 1, 1)
        cross_v = cross_v.reshape(
            -1, self.cross_size * self.cross_size, 1, 1)
        high = high + self.drop_path(self.attn_h(high, cross_k, cross_v))
        high = high + self.drop_path(self.mlp_h(high))

        if self.use_injection:
            high_down = self.high_to_low(high)
            if high_down.shape[2:] != low.shape[2:]:
                high_down = resize(
                    high_down,
                    size=low.shape[2:],
                    mode='bilinear',
                    align_corners=False)
            low = low + high_down

        return high, low


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
class STDCResNetCRA(BaseSegmentor):
    """STDC detail branch + ResNet semantic branch with CRA interaction."""

    def __init__(self,
                 num_classes,
                 stdc_base=64,
                 resnet_cfg=None,
                 fusion_channels=128,
                 num_heads=8,
                 cross_size=12,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_aux_head=True,
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
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.ignore_index = ignore_index
        self.use_aux_head = use_aux_head
        self.pretrained = pretrained
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.stdc = ShallowNet_lk(
            base=stdc_base,
            in_channels=in_channels,
            num_classes=num_classes)
        stdc_x4_channels = stdc_base
        stdc_x8_channels = stdc_base * 4
        stdc_x16_channels = stdc_base * 8

        if resnet_cfg is None:
            resnet_cfg = dict(
                type='ResNetV1c',
                depth=18,
                in_channels=stdc_x4_channels,
                stem_channels=64,
                base_channels=64,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True)
        self.resnet = build_backbone(resnet_cfg)

        res_base_channels = resnet_cfg.get('base_channels', 64)
        expansion = self.resnet.block.expansion
        res_x16_channels = res_base_channels * expansion
        res_x32_channels = res_base_channels * 8 * expansion

        self.cra8 = CrossResolutionAttention(
            high_channels=stdc_x8_channels,
            low_channels=res_x16_channels,
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            cross_size=cross_size)
        self.cra16 = CrossResolutionAttention(
            high_channels=stdc_x16_channels,
            low_channels=res_x32_channels,
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            cross_size=cross_size)

        norm_cfg = resnet_cfg.get('norm_cfg', dict(type='BN', requires_grad=True))
        act_cfg = dict(type='ReLU')
        self.proj_h8 = ConvModule(
            stdc_x8_channels,
            fusion_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.proj_h16 = ConvModule(
            stdc_x16_channels,
            fusion_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.proj_r16 = ConvModule(
            res_x16_channels,
            fusion_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.proj_r32 = ConvModule(
            res_x32_channels,
            fusion_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.fuse16 = ConvModule(
            fusion_channels * 3,
            fusion_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.fuse8 = ConvModule(
            fusion_channels * 2,
            fusion_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.decode_head = SegHead(fusion_channels, fusion_channels,
                                   num_classes)
        if self.use_aux_head:
            self.auxiliary_head = SegHead(fusion_channels, fusion_channels,
                                          num_classes)

        self.loss_decode = build_loss(loss_decode)
        self.loss_aux = build_loss(loss_aux) if loss_aux is not None else None

    def init_weights(self):
        self.apply(init_weights_kaiming)
        if isinstance(self.pretrained, str):
            from mmcv.runner import load_checkpoint
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    def _resnet_stem(self, x):
        if self.resnet.deep_stem:
            return self.resnet.stem(x)
        x = self.resnet.conv1(x)
        x = self.resnet.norm1(x)
        x = self.resnet.relu(x)
        return x

    def _forward_features(self, x):
        x2 = self.stdc.x2(x)
        s4 = self.stdc.x4(x2)

        h8 = self.stdc.x8(s4)

        r8 = self._resnet_stem(s4)
        r16 = self.resnet.maxpool(r8)
        r16 = self.resnet.layer1(r16)
        h8, r16 = self.cra8(h8, r16)

        h16 = self.stdc.x16(h8)
        r32 = self.resnet.layer2(r16)
        r32 = self.resnet.layer3(r32)
        r32 = self.resnet.layer4(r32)
        h16, r32 = self.cra16(h16, r32)

        r32_16 = resize(
            self.proj_r32(r32),
            size=h16.shape[2:],
            mode='bilinear',
            align_corners=False)
        sem16 = self.fuse16(
            torch.cat([self.proj_h16(h16), self.proj_r16(r16), r32_16],
                      dim=1))

        sem8 = resize(
            sem16,
            size=h8.shape[2:],
            mode='bilinear',
            align_corners=False)
        out8 = self.fuse8(torch.cat([self.proj_h8(h8), sem8], dim=1))
        return out8, sem16, (h8, h16, r16, r32)

    def extract_feat(self, img):
        return self._forward_features(img)

    def _forward_logits(self, x, return_aux=False):
        out8, sem16, features = self._forward_features(x)
        logits = [self.decode_head(out8)]
        if return_aux and self.use_aux_head:
            logits.append(self.auxiliary_head(sem16))
        logits = [
            resize(
                logit,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logits
        ]
        if return_aux:
            return logits, features
        return logits[0]

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
        logits, _ = self._forward_logits(img, return_aux=True)
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
        return self._forward_logits(img, return_aux=False)

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
