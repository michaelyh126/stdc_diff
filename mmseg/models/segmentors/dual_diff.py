import torch
import time
from torch import nn
#from ..decode_heads.lpls_utils import Lap_Pyramid_Conv
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from .encoder_decoder_pid import EncoderDecoderPid
from mmcv.runner import auto_fp16
from mmseg.models.decode_heads.harr import HarrDown

@SEGMENTORS.register_module()
class DualDiff(EncoderDecoderPid):
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
        super(DualDiff, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        # self.decode_head = nn.ModuleList()
        #self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=1)
        # self.decode_head = builder.build_head(decode_head[0])
        self.refine_head = builder.build_head(decode_head[0])
        # self.diff_head = builder.build_head(decode_head[2])
        # self.harr_down = HarrDown()
        # print(self.decode_head)d
        # print(self.refine_head)
        self.align_corners = self.refine_head.align_corners
        self.num_classes = self.refine_head.num_classes

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # TODO: 下采样图像设定尚未完成
        # 这里值得注意的是，输入图像应该分大分辨率和小分辨率两种
        # 目前的计划：
        # 大分辨率图像：参数传入的image， 输入refine_head中
        # 小分辨率图像：参数传入的image下采样到原来的0.25倍数，输入feature_extractor, 即原有的分支中

        prev_outputs=None
        out = self.refine_head.forward_test(img, prev_outputs, img_metas, self.test_cfg)

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # torch.cuda.synchronize()
        # end_time2 = time.perf_counter()
        # print(end_time2 - start_time2)
        # print("--------------------------")
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
        img_refine=img
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_refine, img_metas, gt_semantic_seg)
        losses.update(loss_decode)


        return losses

    # TODO: 搭建refine的head
    def _decode_head_forward_train(self, x, img, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        prev_features=None
        loss_refine, *loss_contrsative_list = self.refine_head.forward_train(img, prev_features, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_refine, 'refine'))

        j = 1
        for loss_aux in loss_contrsative_list:
            losses.update(add_prefix(loss_aux, 'aux_' + str(j)))
            j += 1
        return losses
