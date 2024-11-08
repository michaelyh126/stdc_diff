import torch
import time
from torch import nn
#from ..decode_heads.lpls_utils import Lap_Pyramid_Conv
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from mmcv.runner import auto_fp16

@SEGMENTORS.register_module()
class StarEncoder(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 down_ratio,
                 backbone,
                 decode_head,
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
        super(StarEncoder, self).__init__(
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
        self.decode_head = builder.build_head(decode_head[0])
        # self.refine_head = builder.build_head(decode_head[1])
        # self.diff_head=builder.build_head(decode_head[2])
        # print(self.decode_head)
        # print(self.refine_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # TODO: 下采样图像设定尚未完成
        # 这里值得注意的是，输入图像应该分大分辨率和小分辨率两种
        # 目前的计划：
        # 大分辨率图像：参数传入的image， 输入refine_head中
        # 小分辨率图像：参数传入的image下采样到原来的0.25倍数，输入feature_extractor, 即原有的分支中
        if self.is_frequency:
            deeplab_inputs = self.lap_prymaid_conv.pyramid_decom(img)[0]
            img_os2 = nn.functional.interpolate(deeplab_inputs, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])
        else:
            img_os2 = nn.functional.interpolate(img, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])

        if self.refine_input_ratio == 1.:
            img_refine = img
        elif self.refine_input_ratio < 1.:
            img_refine = nn.functional.interpolate(img, size=[int(img.shape[-2] * self.refine_input_ratio), int(img.shape[-1] * self.refine_input_ratio)])
        x = self.extract_feat(img_os2)
        out_g, prev_outputs = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        # out = self.decode_head.forward_test(x , img_metas, self.test_cfg)
        out = resize(
            input=out_g,
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

        import time
        start_forward=time.time()
        # img_os2:将deeplabv3输入的图像size下采样为原来的一半
        if self.is_frequency:
            deeplab_inputs = self.lap_prymaid_conv.pyramid_decom(img)[0]
            img_os2 = nn.functional.interpolate(deeplab_inputs, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])
        else:
            img_os2 = nn.functional.interpolate(img, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])
        start=time.time()
        x = self.extract_feat(img_os2)
        print(f'backbone {time.time() - start}')
        if self.refine_input_ratio == 1.:
            img_refine = img
        elif self.refine_input_ratio < 1.:
            img_refine = nn.functional.interpolate(img, size=[int(img.shape[-2] * self.refine_input_ratio), int(img.shape[-1] * self.refine_input_ratio)])
        losses = dict()
        loss_decode,fm_decoder = self.decode_head.forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        return losses