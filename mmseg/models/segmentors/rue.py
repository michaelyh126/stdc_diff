import torch
import time
from torch import nn
import torch.nn.functional as F
#from ..decode_heads.lpls_utils import Lap_Pyramid_Conv
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from .encoder_decoder_refine import EncoderDecoderRefine
from mmcv.runner import auto_fp16
from mmseg.models.decode_heads.harr import HarrDown
import torch.nn.functional as F
from other_utils.heatmap import visualize_feature_map

import torch


def split_to_grid_patches(img, rows=8, cols=8):
    _, _, H, W = img.shape
    h_step = H // rows
    w_step = W // cols
    patches_container = []
    for i in range(rows):
        for j in range(cols):
            h_start = i * h_step
            h_end = (i + 1) * h_step
            w_start = j * w_step
            w_end = (j + 1) * w_step
            patch = img[:, :, h_start:h_end, w_start:w_end]
            patches_container.append(patch)

    return patches_container


def calculate_entropy(out):
    softmax_out = F.softmax(out, dim=1)
    log_softmax_out = torch.log(softmax_out + 1e-10)
    product = softmax_out * log_softmax_out
    M = out.shape[1]
    entropy = -torch.sum(product, dim=1) / torch.log(torch.tensor(M).float())

    return entropy

def select( estimation, n=8, k_rate=0.5):
    estimation = F.interpolate(estimation, size=(n, n), mode='bilinear', align_corners=True)
    flat_estimation = estimation.view(estimation.size(0), estimation.size(1), -1)
    k=int(k_rate*n*n)
    _, topk_indices = torch.topk(flat_estimation, k, dim=2)
    # rows = topk_indices // 10
    # cols = topk_indices % 10
    return topk_indices


@SEGMENTORS.register_module()
class RUE(EncoderDecoderRefine):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 down_ratio=4,
                 backbone=None,
                 decode_head=None,
                 refine_input_ratio=1.,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 is_frequency=False,
                 pretrained=None,
                 init_cfg=None):
        self.num_stages = num_stages
        self.is_frequency = is_frequency
        self.down_scale = down_ratio
        self.refine_input_ratio = refine_input_ratio

        super(RUE, self).__init__(
            num_stages=num_stages,
            down_ratio=down_ratio,
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.conv1=nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        # self.decode_head = nn.ModuleList()
        #self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=1)
        self.decode_head = builder.build_head(decode_head[0])
        self.refine_head = builder.build_head(decode_head[1])
        # self.distill_head = builder.build_head(decode_head[1])
        # self.diff_head = builder.build_head(decode_head[2])
        # self.harr_down = HarrDown()
        # print(self.decode_head)d
        # print(self.refine_head)
        self.align_corners = self.refine_head.align_corners
        self.num_classes = self.refine_head.num_classes

    def estimate(self,out, f_d1, f_d2, a=0.5):
        entropy = calculate_entropy(out)
        entropy=entropy.unsqueeze(1)
        entropy = F.interpolate(entropy, size=f_d1.shape[2:], mode='bilinear', align_corners=True)

        f_d2_up = F.interpolate(f_d2, size=f_d1.shape[2:], mode='bilinear', align_corners=True)
        f_d2_conv =self.conv1(f_d2_up)
        diff = torch.abs(f_d2_conv - f_d1)
        mean_val = torch.mean(diff, dim=1, keepdim=True)
        max_val, _ = torch.max(diff, dim=1, keepdim=True)
        combined = torch.cat([mean_val, max_val], dim=1)
        f = self.conv2(combined)

        out=(1-a)*f+a*entropy
        return out



    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # TODO: 下采样图像设定尚未完成
        # 这里值得注意的是，输入图像应该分大分辨率和小分辨率两种
        # 目前的计划：
        # 大分辨率图像：参数传入的image， 输入refine_head中
        # 小分辨率图像：参数传入的image下采样到原来的0.25倍数，输入feature_extractor, 即原有的分支中
        prev_outputs = None
        img_os2 = nn.functional.interpolate(img, size=[img.shape[-2] // self.down_scale,
                                                       img.shape[-1] // self.down_scale])
        x = self.extract_feat(img_os2)
        out, prev_outputs = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        estimation=self.estimate(out,x[0],x[1])
        patch_index=select(estimation)
        out = self.refine_head.forward_test(prev_outputs[0], img, patch_index,estimation, img_metas, self.test_cfg)

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        return out


    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # img_os2:将deeplabv3输入的图像size下采样为原来的一半
        x=None
        img_os2 = nn.functional.interpolate(img, size=[img.shape[-2] // self.down_scale,
                                                       img.shape[-1] // self.down_scale])
        x = self.extract_feat(img_os2)
        out, prev_outputs = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        estimation=self.estimate(out,x[0],x[1])
        patch_index=select(estimation)
        losses = dict()
        loss_decode = self._decode_head_forward_train(prev_outputs[0], img, patch_index,estimation, img_metas, gt_semantic_seg)
        losses.update(loss_decode)


        return losses

    # TODO: 搭建refine的head
    def _decode_head_forward_train(self, fd, img, patch_index,estimation, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        # loss_refine, *loss_contrsative_list = self.refine_head.forward_train(img, prev_features, img_metas, gt_semantic_seg, self.train_cfg)
        loss_refine,*loss_contrsative_list = self.refine_head.forward_train(fd, img, patch_index,estimation, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_refine, 'refine'))

        j = 1
        for loss_aux in loss_contrsative_list:
            losses.update(add_prefix(loss_aux, 'aux_' + str(j)))
            j += 1
        return losses
