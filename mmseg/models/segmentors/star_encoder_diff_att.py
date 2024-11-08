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
from mmseg.models.decode_heads.diff_fusion import FeatureFusionModule
from mmseg.models.decode_heads.stdc_head import ShallowNet
from other_utils.heatmap import save_image,save_heatmap
from mmseg.models.decode_heads.isdhead import Lap_Pyramid_Conv
import torch.nn.functional as F

@SEGMENTORS.register_module()
class StarEncoderDiffAtt(EncoderDecoder):
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

        super(StarEncoderDiffAtt, self).__init__(
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
        self.diff_head=builder.build_head(decode_head[1])
        self.stdc_net=builder.build_head(decode_head[2])
        self.diff_fusion=builder.build_head(decode_head[3])
        # self.stdc_net = ShallowNet(in_channels=6, pretrain_model='')
        # self.diff_fusion=FeatureFusionModule(in_chan=384, out_chan=128)
        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)

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
        diff_pred = self.diff_head.forward_test_diff(prev_outputs, img_metas, self.test_cfg)


        prymaid_results = self.lap_prymaid_conv.pyramid_decom(img)
        high_residual_1 = prymaid_results[0]
        high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
                                        align_corners=False)
        high_residual_input = torch.cat([high_residual_1, high_residual_2], dim=1)
        feat8, feat16=self.stdc_net.forward_stdc_test(high_residual_input)
        shallow_feat=feat16
        deep_feat=prev_outputs[0]
        diff_pred = resize(
            input=diff_pred,
            size=shallow_feat.shape[2:],
            mode='bilinear',
            align_corners=None)
        mask = diff_pred

        # channel_data = diff_pred[:, 1, :, :]
        # # 计算每个批次的最大值和最小值
        # max_vals = torch.max(channel_data.view(channel_data.size(0), -1), dim=1).values
        # min_vals = torch.min(channel_data.view(channel_data.size(0), -1), dim=1).values
        # threshold=2*(max_vals-min_vals)/3+min_vals
        # thresholds_expanded = threshold.view(-1, 1, 1)
        # mask = (channel_data > thresholds_expanded).float()
        # mask = mask.unsqueeze(1)
        out_g = self.diff_fusion.forward_diff_fusion_test(shallow_feat, deep_feat, mask)

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
        loss_decode,prev_features = self.decode_head.forward_train(x, img_metas, gt_semantic_seg)
        losses.update(add_prefix(loss_decode, 'decode'))
        loss_diff,diff_map, diff_pred = self.diff_head.forward_train_diff(prev_features, img_metas, gt_semantic_seg,
                                                                 self.train_cfg)
        losses.update(add_prefix(loss_diff, 'diff'))

        prymaid_results = self.lap_prymaid_conv.pyramid_decom(img)
        high_residual_1 = prymaid_results[0]
        high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
                                        align_corners=False)
        high_residual_input = torch.cat([high_residual_1, high_residual_2], dim=1)

        loss_stdc,feat8, feat16 = self.stdc_net.forward_stdc_train(high_residual_input,img_metas,gt_semantic_seg)
        # TODO
        # STDC损失
        # losses.update(add_prefix(loss_stdc, 'stdc'))

        shallow_feat=feat16
        deep_feat=prev_features[0]
        diff_pred = resize(
            input=diff_pred,
            size=shallow_feat.shape[2:],
            mode='bilinear',
            align_corners=None)
        mask=diff_pred
        # mask=torch.argmax(diff_pred,dim=1).unsqueeze(1)
        # channel_data = diff_pred[:, 1, :, :]
        # # 计算每个批次的最大值和最小值
        # max_vals = torch.max(channel_data.view(channel_data.size(0), -1), dim=1).values
        # min_vals = torch.min(channel_data.view(channel_data.size(0), -1), dim=1).values
        # threshold=2*(max_vals-min_vals)/3+min_vals
        # thresholds_expanded = threshold.view(-1, 1, 1)
        # mask = (channel_data > thresholds_expanded).float()
        # mask = mask.unsqueeze(1)

        # mask_save = resize(
        #     input=mask,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=None)
        # mask_save=mask_save.squeeze(1)
        # save_image(mask_save[1].detach().cpu().numpy(),filename='mask',save_dir='D:\deep_learning\ISDNetV2\diff_dir\pred')

        loss_fusion, seg_logit=self.diff_fusion.forward_diff_fusion_train(shallow_feat,deep_feat,mask,img_metas, gt_semantic_seg)
        losses.update(add_prefix(loss_fusion, 'fusion'))
        return losses
