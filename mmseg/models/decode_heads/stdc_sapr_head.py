import torch.nn.functional as F

from ..builder import HEADS
from .stdc import MyStdcHead


@HEADS.register_module()
class MyStdcSAPRHead(MyStdcHead):
    """STDC head variant that exposes the fusion feature for SAPR training.

    The original ``MyStdcHead`` is intentionally left untouched so other STDC
    experiments keep exactly the same behavior.
    """

    def extract_sapr_feature(self, inputs):
        shallow_feat8, shallow_feat16 = self.stdc_net(inputs)
        shallow_feat16 = self.convert_shallow16(shallow_feat16)
        shallow_feat8 = self.convert_shallow8(shallow_feat8)
        _, _, h, w = shallow_feat8.size()
        shallow_feat16 = F.interpolate(
            shallow_feat16,
            size=(h, w),
            mode='bilinear',
            align_corners=False)
        fusion = self.addConv(shallow_feat8, shallow_feat16)
        return fusion, shallow_feat8

    def forward_with_feature(self,
                             inputs,
                             prev_output=None,
                             train_flag=True,
                             mask=None,
                             gt=None,
                             img_metas=None,
                             train_cfg=None,
                             diff_pred_deep=None):
        fusion, shallow_feat8 = self.extract_sapr_feature(inputs)
        output = self.cls_seg(fusion)
        if train_flag:
            aux_output = self.cls_seg8(shallow_feat8)
            return output, aux_output, fusion
        return output, fusion

    def forward(self,
                inputs,
                prev_output,
                train_flag=True,
                mask=None,
                gt=None,
                img_metas=None,
                train_cfg=None,
                diff_pred_deep=None):
        if train_flag:
            output, aux_output, _ = self.forward_with_feature(
                inputs, prev_output, train_flag, mask, gt, img_metas,
                train_cfg, diff_pred_deep)
            return output, aux_output
        output, _ = self.forward_with_feature(
            inputs, prev_output, train_flag, mask, gt, img_metas, train_cfg,
            diff_pred_deep)
        return output
