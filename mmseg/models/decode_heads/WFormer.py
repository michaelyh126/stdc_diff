import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead


class HaarDWT2D(nn.Module):
    """Orthogonal Haar discrete wavelet transform."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        _, _, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, \
            f'H and W must be even, got H={h}, W={w}'

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return torch.cat([ll, lh, hl, hh], dim=1)


class HaarIWT2D(nn.Module):
    """Inverse orthogonal Haar discrete wavelet transform."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c4, h, w = x.shape
        assert c4 % 4 == 0, f'Channel must be divisible by 4, got {c4}'
        c = c4 // 4

        ll, lh, hl, hh = torch.chunk(x, 4, dim=1)

        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5

        out = x.new_zeros(b, c, h * 2, w * 2)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out


def patchify(x, patch_size):
    """Split [B, C, H, W] into non-overlapping patches."""

    b, c, h, w = x.shape
    p = patch_size
    assert h % p == 0 and w % p == 0, \
        f'H/W must be divisible by patch_size={p}, got H={h}, W={w}'

    hp = h // p
    wp = w // p

    patches = F.unfold(x, kernel_size=p, stride=p)
    patches = patches.transpose(1, 2).contiguous()
    patches = patches.view(b, hp * wp, c, p, p)
    return patches, hp, wp


def unpatchify(patches, hp, wp):
    """Restore non-overlapping patches back to feature map."""

    b, n, c, p, _ = patches.shape
    assert n == hp * wp, f'N mismatch: N={n}, Hp*Wp={hp * wp}'

    x = patches.view(b, n, c * p * p).transpose(1, 2).contiguous()
    return F.fold(x, output_size=(hp * p, wp * p), kernel_size=p, stride=p)


def window_partition(x, window_size):
    """Partition [B, C, H, W] into non-overlapping windows."""

    b, c, h, w = x.shape
    win_h, win_w = window_size

    pad_h = (win_h - h % win_h) % win_h
    pad_w = (win_w - w % win_w) % win_w
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    hp, wp = x.shape[2], x.shape[3]
    x = x.view(b, c, hp // win_h, win_h, wp // win_w, win_w)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = x.view(-1, c, win_h, win_w)
    return windows, (h, w), (hp, wp)


def window_reverse(windows, window_size, orig_hw, padded_hw, batch_size):
    """Reverse window partition back to [B, C, H, W]."""

    win_h, win_w = window_size
    orig_h, orig_w = orig_hw
    padded_h, padded_w = padded_hw
    c = windows.shape[1]

    x = windows.view(
        batch_size, padded_h // win_h, padded_w // win_w, c, win_h, win_w)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(batch_size, c, padded_h, padded_w)
    return x[:, :, :orig_h, :orig_w]


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GlobalSelfAttention(nn.Module):
    """Global self-attention over the whole token sequence."""

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=True,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, \
            f'dim={dim} must be divisible by num_heads={num_heads}'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, l, c = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(b, l, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, l, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class WFormerBlockNoGroup(nn.Module):
    """Wavelet transformer block following the paper's Figure 2 idea."""

    def __init__(self,
                 in_channels,
                 embed_dim,
                 image_size=None,
                 patch_size=8,
                 token_stride=8,
                 num_heads=8,
                 mlp_ratio=4.0,
                 drop=0.0,
                 attn_drop=0.0):
        super().__init__()

        assert embed_dim % 4 == 0, 'embed_dim must be divisible by 4'
        assert patch_size % 2 == 0, 'patch_size must be even'

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.token_stride = token_stride
        self.base_pos_size = self._build_base_pos_size(image_size, patch_size)
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        self.pre_proj = nn.Conv2d(
            in_channels, embed_dim // 4, kernel_size=1, stride=1, padding=0)
        self.dwt = HaarDWT2D()
        self.iwt = HaarIWT2D()

        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, self.base_pos_size[0], self.base_pos_size[1]))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Common transformer reduction for no-group training:
        # reduce the query grid before attention, while keeping
        # the paper-style wavelet reduction for K/V.
        self.query_down = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=token_stride, stride=token_stride)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop)
        self.query_up = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.post_proj = nn.Conv2d(
            embed_dim // 4, in_channels, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _build_base_pos_size(image_size, patch_size):
        if image_size is None:
            return (1, 1)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert len(image_size) == 2, \
            'image_size must be int, tuple/list of length 2, or None'

        base_h = ((image_size[0] + patch_size - 1) // patch_size) * patch_size // 2
        base_w = ((image_size[1] + patch_size - 1) // patch_size) * patch_size // 2
        return (max(base_h, 1), max(base_w, 1))

    def _pad_to_even(self, x):
        h, w = x.shape[2], x.shape[3]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return x

    def _wavelet_reduce_tokens(self, x):
        x = self._pad_to_even(x)
        x = self.dwt(x)
        return x.flatten(2).transpose(1, 2).contiguous()

    def forward(self, x):
        _, c, h, w = x.shape

        assert c == self.in_channels, \
            f'channel mismatch: got {c}, expected {self.in_channels}'

        shortcut = x
        x = self.pre_proj(x)
        x = self._pad_to_even(x)
        x = self.dwt(x)
        x_h, x_w = x.shape[2], x.shape[3]

        pos_embed = F.interpolate(
            self.pos_embed,
            size=(x_h, x_w),
            mode='bilinear',
            align_corners=False)
        x = x + pos_embed

        q_grid = self.query_down(x)
        q_h, q_w = q_grid.shape[2], q_grid.shape[3]
        q_tokens = q_grid.flatten(2).transpose(1, 2).contiguous()
        q_tokens = self.norm1(q_tokens)

        q = self.q_proj(q_tokens)
        k_tokens = self._wavelet_reduce_tokens(self.k_proj(q_grid))
        v_tokens = self._wavelet_reduce_tokens(self.v_proj(q_grid))

        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k_tokens.view(k_tokens.shape[0], k_tokens.shape[1], self.num_heads,
                          self.head_dim).transpose(1, 2)
        v = v_tokens.view(v_tokens.shape[0], v_tokens.shape[1], self.num_heads,
                          self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(q_tokens.shape[0], q_tokens.shape[1], self.embed_dim)
        out = self.proj(out)
        out = self.proj_drop(out)

        out = out + q_tokens
        out = out + self.mlp(self.norm2(out))

        out = out.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, q_h, q_w)
        out = self.query_up(out)
        out = F.interpolate(out, size=(x_h, x_w), mode='bilinear', align_corners=False)
        x = x + out

        x = self.iwt(x)
        x = self.post_proj(x)
        return x[:, :, :h, :w] + shortcut


class WFormerStageNoGroup(nn.Module):
    """Stack multiple wavelet transformer blocks."""

    def __init__(self,
                 in_channels,
                 embed_dim,
                 image_size=None,
                 depth=2,
                 patch_size=8,
                 token_stride=8,
                 num_heads=8,
                 mlp_ratio=4.0,
                 drop=0.0,
                 attn_drop=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            WFormerBlockNoGroup(
                in_channels=in_channels,
                embed_dim=embed_dim,
                image_size=image_size,
                patch_size=patch_size,
                token_stride=token_stride,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


@HEADS.register_module()
class WFormerHead(BaseCascadeDecodeHead):
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 down_ratio,
                 prev_channels,
                 img_size=None,
                 reduce=False,
                 decoder_flag='aff',
                 embed_dim=None,
                 depth=2,
                 patch_size=8,
                 token_stride=8,
                 num_heads=8,
                 mlp_ratio=4.0,
                 drop=0.0,
                 attn_drop=0.0,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal',
                     std=0.01,
                     override=dict(name='conv_seg'))):
        super(WFormerHead, self).__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            in_index=in_index,
            input_transform=input_transform,
            loss_decode=loss_decode,
            ignore_index=ignore_index,
            sampler=sampler,
            align_corners=align_corners,
            init_cfg=init_cfg)
        del down_ratio, prev_channels, reduce, decoder_flag

        self.img_size = img_size
        self.embed_dim = embed_dim or self.channels
        self.depth = depth

        self.input_proj = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.channels,
            kernel_size=1,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.wavelet_stage = WFormerStageNoGroup(
            in_channels=self.channels,
            embed_dim=self.embed_dim,
            image_size=self.img_size,
            depth=depth,
            patch_size=patch_size,
            token_stride=token_stride,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop)

        self.aux_head = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self,
                inputs,
                prev_output,
                train_flag=True,
                mask=None,
                gt=None,
                img_metas=None,
                train_cfg=None,
                diff_pred_deep=None):
        del prev_output, mask, gt, img_metas, train_cfg, diff_pred_deep

        feat = self.input_proj(inputs)
        feat = self.wavelet_stage(feat)
        output = self.cls_seg(feat)

        if train_flag:
            aux_output = self.aux_head(feat)
            return output, aux_output
        return output

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg, mask=None):
        output, aux_output = self.forward(
            inputs,
            prev_output,
            train_flag=True,
            mask=mask,
            gt=gt_semantic_seg,
            img_metas=img_metas,
            train_cfg=train_cfg,
            diff_pred_deep=mask)
        losses = self.losses(output, gt_semantic_seg)
        losses_aux = self.losses(aux_output, gt_semantic_seg)
        return losses, losses_aux

    def forward_test(self, inputs, prev_output, img_metas, test_cfg, mask=None):
        del img_metas, test_cfg, mask
        return self.forward(inputs, prev_output, train_flag=False)
