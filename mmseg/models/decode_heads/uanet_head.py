import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from .pid import AddFuse
from .diff_head import DiffHead
from mmseg.models.decode_heads.UANet.models.UANet import UANet_Res50





@HEADS.register_module()
class UANet_head(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels,img_size, reduce=False,decoder_flag='aff', **kwargs):
        super(UANet_head, self).__init__(**kwargs)
        self.decoder_flag=decoder_flag
        self.shallow_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
                type='BCEDiceLoss'))
        self.down_ratio = down_ratio
        self.convert_shallow8=nn.Conv2d(256,self.channels,stride=1,kernel_size=1)
        self.uanet=UANet_Res50(32,2)
        # self.stdc_net = ShallowNet_rf63(in_channels=3, pretrain_model="/root/autodl-tmp/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        # self.stdc_net = ShallowNet(in_channels=3, pretrain_model="/root/autodl-tmp/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        # self.stdc_net = ShallowNet_lk(in_channels=3,num_classes=self.num_classes)
        # self.stdc_net = ShallowNet_rep(in_channels=3, pretrain_model="/root/autodl-tmp/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        self.reduce = Reducer() if reduce else None
        self.convert_shallow16=nn.Conv2d(512,self.channels,stride=1,kernel_size=3,padding=1)
        self.addConv=AddFuse(self.channels,self.channels)
        self.fuse=AddFuse(self.channels,self.channels)
        self.cls_seg8 = nn.Conv2d(128, 7, kernel_size=1)





    def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
        """Forward function."""
        # if train_flag==False:
        #     self.stdc_net.switch_to_deploy()
        # shallow_feat8, shallow_feat16 = self.stdc_net(inputs)
        f5,f4,f3,f2,f1 = self.uanet(inputs)
        if train_flag==True:
            return f1,f2,f3,f4,f5
        else:
            return f1







    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg,mask=None):
        f1,f2,f3,f4,f5 = self.forward(
            inputs, prev_output,
            mask=mask,
            gt=gt_semantic_seg,
            img_metas=img_metas,
            train_cfg=train_cfg,
            diff_pred_deep=mask)
        losses = self.losses(f1, gt_semantic_seg)
        losses_f2 = self.losses(f2, gt_semantic_seg)
        losses_f3 = self.losses(f3, gt_semantic_seg)
        losses_f4 = self.losses(f4, gt_semantic_seg)
        losses_f5 = self.losses(f5, gt_semantic_seg)
        return  losses,losses_f2,losses_f3,losses_f4,losses_f5



    def forward_test(self, inputs, prev_output, img_metas, test_cfg,mask=None):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """

        return self.forward(inputs, prev_output, False,diff_pred_deep=mask)

