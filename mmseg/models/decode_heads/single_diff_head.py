import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from .diff_fusion import FeatureFusionModule
from .harr import HarrUp
from other_utils.heatmap import save_image,save_heatmap
from mmseg.ops import resize
from mmseg.models.losses.detail_loss import DetailAggregateLoss
from .pid import Bag,AdaptiveFrequencyFusion,AddFuse
from .diff_head import DiffHead
from .diff_point import DiffPoint
from .spnet import SpNet,get_coordsandfeatures,SparseResNet50
from .spnetv2 import SpNetV2
from other_utils.split_tensor import split_tensor,restore_tensor
from .pid import segmenthead
from .sdd_stdc_head import ShallowNet
from .pidnet_single import PIDNet
# from .pidnet import PIDNet
# from .pidnet_un import PIDNet
# from .pidnet_distill import PIDNet
# from .pidnet_stdc import PIDNet
from mmseg.models.losses.bce_dice_loss import BCEDiceLoss
from mmseg.models.losses.detail_loss import DetailAggregateLoss
from .isdhead import RelationAwareFusion
from mmseg.models.sampler.dysample import DySample
from other_utils.histogram import tensor_histogram



class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=255, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target,name):
        loss=dict()
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        balance_weights = [0.5, 0.5]
        sb_weights = 0.5
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * \
                        (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])

        elif len(score) == 1:
            loss[name+'ohem_loss']=sb_weights * self._ohem_forward(score[0], target)
            return loss

        else:
            raise ValueError("lengths of prediction and target are not identical!")


class SegmentationHead(nn.Module):
    def __init__(self, conv_cfg, norm_cfg, act_cfg, in_channels, mid_channels, n_classes, *args, **kwargs):
        super(SegmentationHead, self).__init__()

        self.conv_bn_relu = ConvModule(in_channels, mid_channels, 3,
                                       stride=1,
                                       padding=1,
                                       conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)

        self.conv_out = nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.conv_out(x)
        return x


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, gauss_chl=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.gauss_chl = gauss_chl
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda')):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(self.gauss_chl, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr






@HEADS.register_module()
class SingleDiffHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels,img_size, reduce=False,decoder_flag='aff', **kwargs):
        super(SingleDiffHead, self).__init__(**kwargs)
        self.decoder_flag=decoder_flag
        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
        self.shallow_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
                type='BCEDiceLoss'))
        self.down_ratio = down_ratio
        self.convert_shallow8=nn.Conv2d(256,self.channels,stride=1,kernel_size=1)
        self.stdc_net = ShallowNet(in_channels=3, pretrain_model="/root/autodl-tmp/pretrained models/STDCNet813M_73.91.tar",num_classes=self.num_classes)
        self.pid=PIDNet(m=2, n=3, num_classes=self.num_classes, planes=32, ppm_planes=96, head_planes=128, augment=True)
        h,w=img_size
        self.ohem=OhemCrossEntropy()
        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
        self.reduce = Reducer() if reduce else None
        self.conv_seg_aux_16 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                                self.channels // 2, self.num_classes, kernel_size=1)
        self.convert_shallow16=nn.Conv2d(512,self.channels,stride=1,kernel_size=3,padding=1)
        self.fuse8 = RelationAwareFusion(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg, ext=2)
        self.shallow_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
                type='BCEDiceLoss'))
        # self.addConv=ConvModule(128,self.channels,stride=1,kernel_size=3,padding=1)
        self.addConv=AddFuse(self.channels,self.channels)
        self.aff=AdaptiveFrequencyFusion(sp_channels=256,co_channels=512,out_channels=128,mid_channels=128,kernel_size=3)
        # self.sampler=DySample(in_channels=self.num_classes,scale=4,groups=1)


    def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
        """Forward function."""
        decoder_flag=self.decoder_flag

        shallow_feat8, shallow_feat16 = self.stdc_net(inputs)


        # add fusion
        # shallow_feat16 = self.convert_shallow16(shallow_feat16)
        # shallow_feat8=self.convert_shallow8(shallow_feat8)
        # _, _, h, w = shallow_feat8.size()
        # shallow_feat16 = F.interpolate(shallow_feat16, size=(h , w ), mode='bilinear', align_corners=False)
        # fusion=self.addConv(shallow_feat8,shallow_feat16)

        # raf fusion
        # shallow_feat16 = self.convert_shallow16(shallow_feat16)
        # _, aux_feat8, fusion = self.fuse8(shallow_feat8, shallow_feat16)

        # freq fusion
        fusion=self.aff(shallow_feat8,shallow_feat16)

        output = self.cls_seg(fusion)


        # predict = self.pid.forward_sp(fusion,output)
        feats=[]
        if decoder_flag==1:
            if train_flag:
                predict = self.pid.forward_dual(fusion, output)
                loss_shallow_diff, diff_map, diff_pred_shallow = self.shallow_diff.forward_train_diff(predict, img_metas,
                                                                                                      gt, train_cfg)
                diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
                                           align_corners=self.align_corners)
                diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
                diff_pred_shallow_sig = diff_pred_shallow
                diff_pred_shallow = (diff_pred_shallow > 0.5).float()
                # gt[diff_pred_shallow == 0] = 255
                output_aux16 = self.conv_seg_aux_16(aux_feat8)
                return output,output_aux16,predict,feats ,loss_shallow_diff,diff_pred_shallow
            else:
                predict = self.pid.forward_dual(fusion, output)
                return predict
        elif decoder_flag=='aff+ohem':
            if train_flag:
                predict = self.pid.forward_dual(fusion, output)
                return output, predict, feats
            else:
                predict = self.pid.forward_dual(fusion, output)
                # tensor_histogram(predict)
                return predict
        elif decoder_flag=='aff':
            if train_flag:
                predict = self.pid.forward_dual(fusion, output)
                loss_shallow_diff, diff_map, diff_pred_shallow,_ = self.shallow_diff.forward_train_diff(predict, img_metas,
                                                                                                      gt, train_cfg)
                diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
                                           align_corners=self.align_corners)
                diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
                diff_pred_shallow_sig = diff_pred_shallow
                diff_pred_shallow = (diff_pred_shallow > 0.5).float()
                return output,predict,feats ,loss_shallow_diff,diff_pred_shallow
            else:
                predict = self.pid.forward_dual(fusion, output)
                # tensor_histogram(predict)
                return predict
        elif decoder_flag=='aff+un':
            if train_flag:
                predict = self.pid.forward_dual(fusion, output)
                loss_shallow_diff, diff_map, diff_pred_shallow,diff_gt = self.shallow_diff.forward_train_diff(predict, img_metas,
                                                                                                      gt, train_cfg)
                diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
                                           align_corners=self.align_corners)
                diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
                diff_pred_shallow_sig = diff_pred_shallow
                diff_pred_shallow = (diff_pred_shallow > 0.5).float()
                return output,predict,feats ,loss_shallow_diff,diff_pred_shallow,diff_gt
            else:
                predict = self.pid.forward_dual(fusion, output)
                # tensor_histogram(predict)
                return predict
        elif decoder_flag=='lossweight':
            if train_flag:
                predict = self.pid.forward_dual(fusion, output)
                loss_shallow_diff, diff_map, diff_pred_shallow = self.shallow_diff.forward_train_diff(predict, img_metas,
                                                                                                      gt, train_cfg)
                diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
                                           align_corners=self.align_corners)
                diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
                diff_pred_shallow_sig = diff_pred_shallow
                diff_pred_shallow = (diff_pred_shallow > 0.5).float()
                return output,predict,feats ,loss_shallow_diff,diff_pred_shallow
            else:
                predict = self.pid.forward_dual(fusion, output)
                # tensor_histogram(predict)
                return predict
        elif decoder_flag=='add':
            if train_flag:
                predict = self.pid.forward_dual(fusion, output)
                return output,predict,feats
            else:
                predict = self.pid.forward_dual(fusion, output)
                # tensor_histogram(predict)
                return predict
        elif decoder_flag=='stdc':
            if train_flag:
                return output,feats
            else:
                return output
        elif decoder_flag=='aff+dysample':
            if train_flag:
                predict = self.pid.forward_dual(fusion, output)
                # predict=self.sampler(predict)
                loss_shallow_diff, diff_map, diff_pred_shallow = self.shallow_diff.forward_train_diff(predict, img_metas,
                                                                                                      gt, train_cfg)
                diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
                                           align_corners=self.align_corners)
                diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
                diff_pred_shallow_sig = diff_pred_shallow
                diff_pred_shallow = (diff_pred_shallow > 0.5).float()
                return output,predict,feats ,loss_shallow_diff,diff_pred_shallow
            else:
                predict = self.pid.forward_dual(fusion, output)
                # predict = self.sampler(predict)
                return predict
        elif decoder_flag=='affwithoutce':
            if train_flag:
                predict = self.pid.forward_dual(fusion, output)
                return output,predict,feats
            else:
                predict = self.pid.forward_dual(fusion, output)
                return predict
        elif decoder_flag==2:
            if train_flag:
                return
            else:
                return
        elif decoder_flag==3:
            if train_flag:
                predict = self.pid.forward_dual(fusion, output)
                return output,predict,feats
            else:
                predict = self.pid.forward_dual(fusion, fusion)
                return predict
        elif decoder_flag==4:
            if train_flag:
                output_aux16 = self.conv_seg_aux_16(aux_feat8)
                return output,output_aux16,feats
            else:
                return output
        else:
            if train_flag:
                predict = self.pid.forward_dual(fusion, output)
                output_aux16 = self.conv_seg_aux_16(aux_feat8)
                return output, predict,output_aux16, feats
            else:
                predict = self.pid.forward_dual(fusion, output)
                return predict



    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg,mask=None):
        if self.decoder_flag==1:
            output,output_aux16, predict,feats,loss_shallow_diff,diff_pred_shallow  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            # predict = F.interpolate(predict, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            # x_extra_p = F.interpolate(x_extra_p, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            # # losses_boundary = self.detail_loss(x8_cls, gt_semantic_seg.squeeze(1))
            # losses_p = self.ohem(x_extra_p, gt_semantic_seg.squeeze(1), 'p_')
            # losses_p['p_ohem_loss']=losses_p['p_ohem_loss']*0.4
            # losses = self.ohem(predict, gt_semantic_seg.squeeze(1), 'middle_')
            losses=self.losses(predict,gt_semantic_seg)
            # losses_p=self.losses(x_extra_p,gt_semantic_seg)
            losses_stdc=self.losses(output,gt_semantic_seg)
            losses_stdc_aux=self.losses(output_aux16,gt_semantic_seg)
            gt_un=gt_semantic_seg.clone()
            gt_un[diff_pred_shallow == 0] = 255
            losses_un=self.losses(predict,gt_un)
            return feats,losses,losses_stdc,losses_stdc_aux,loss_shallow_diff,losses_un
        elif self.decoder_flag=='aff':
            output, predict,feats,loss_shallow_diff,diff_pred_shallow  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses=self.losses(predict,gt_semantic_seg)
            losses_stdc=self.losses(output,gt_semantic_seg)
            gt_un=gt_semantic_seg.clone()
            gt_un[diff_pred_shallow == 0] = 255
            losses_un=self.losses(predict,gt_un)
            return feats,losses,losses_stdc,loss_shallow_diff,losses_un
        elif self.decoder_flag=='aff+un':
            output, predict,feats,loss_shallow_diff,diff_pred_shallow,diff_gt  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses=self.losses(predict,gt_semantic_seg)
            losses_stdc=self.losses(output,gt_semantic_seg)
            gt_un=gt_semantic_seg.clone()
            gt_un[diff_pred_shallow == 0] = 255
            gt_un[diff_gt==0]=255
            losses_un=self.losses(predict,gt_un)
            return feats,losses,losses_stdc,loss_shallow_diff,losses_un
        elif self.decoder_flag=='lossweight':
            output, predict,feats,loss_shallow_diff,diff_pred_shallow  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses=self.losses(predict,gt_semantic_seg)
            losses_stdc=self.losses(output,gt_semantic_seg)
            gt_un=gt_semantic_seg.clone()
            gt_un[diff_pred_shallow == 0] = 255
            losses_un=self.losses(predict,gt_un)
            losses_stdc['loss_seg']=0.4*losses_stdc['loss_seg']
            losses_un['loss_seg']=3*losses_un['loss_seg']
            return feats,losses,losses_stdc,loss_shallow_diff,losses_un
        elif self.decoder_flag=='add':
            output, predict,feats = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses=self.losses(predict,gt_semantic_seg)
            losses_stdc=self.losses(output,gt_semantic_seg)
            return feats,losses,losses_stdc
        elif self.decoder_flag=='stdc':
            output,feats = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses=self.losses(output,gt_semantic_seg)
            return feats,losses
        elif self.decoder_flag=='aff+ohem':
            output, predict,feats  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            predict = F.interpolate(predict, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            output=F.interpolate(output, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            losses=self.ohem(predict, gt_semantic_seg.squeeze(1), 'p_')
            losses_stdc=self.ohem(output, gt_semantic_seg.squeeze(1), 'middle_')
            return feats,losses,losses_stdc
        elif self.decoder_flag=='aff+dysample':
            output, predict,feats,loss_shallow_diff,diff_pred_shallow  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses=self.losses(predict,gt_semantic_seg)
            losses_stdc=self.losses(output,gt_semantic_seg)
            gt_un=gt_semantic_seg.clone()
            gt_un[diff_pred_shallow == 0] = 255
            losses_un=self.losses(predict,gt_un)
            return feats,losses,losses_stdc,loss_shallow_diff,losses_un
        elif self.decoder_flag=='affwithoutce':
            output, predict,feats  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses=self.losses(predict,gt_semantic_seg)
            losses_stdc=self.losses(output,gt_semantic_seg)
            return feats,losses,losses_stdc
        elif self.decoder_flag==2:
            x_extra_p, predict,output,loss_diff1  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            predict = F.interpolate(predict, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            x_extra_p = F.interpolate(x_extra_p, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            output=F.interpolate(output, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            # losses_boundary = self.detail_loss(x8_cls, gt_semantic_seg.squeeze(1))
            losses_p = self.ohem(x_extra_p, gt_semantic_seg.squeeze(1), 'p_')
            losses_p['p_ohem_loss']=losses_p['p_ohem_loss']*0.4
            losses_middle = self.ohem(predict, gt_semantic_seg.squeeze(1), 'middle_')
            losses_middle['middle_ohem_loss']=losses_middle['middle_ohem_loss']*0.5
            losses = self.ohem(output, gt_semantic_seg.squeeze(1), 'output_')
            feats=[]
            return feats,losses,losses_middle,losses_p,loss_diff1
            return

        elif self.decoder_flag==3:
            output, predict,feats  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            # predict = F.interpolate(predict, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            # x_extra_p = F.interpolate(x_extra_p, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            # # losses_boundary = self.detail_loss(x8_cls, gt_semantic_seg.squeeze(1))
            # losses_p = self.ohem(x_extra_p, gt_semantic_seg.squeeze(1), 'p_')
            # losses_p['p_ohem_loss']=losses_p['p_ohem_loss']*0.4
            # losses = self.ohem(predict, gt_semantic_seg.squeeze(1), 'middle_')
            losses=self.losses(predict,gt_semantic_seg)
            # losses_p=self.losses(x_extra_p,gt_semantic_seg)
            losses_stdc=self.losses(output,gt_semantic_seg)
            # losses_stdc_aux=self.losses(output_aux16,gt_semantic_seg)
            return feats,losses,losses_stdc
        elif self.decoder_flag==4:
            output, output_aux16,feats  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses_aux=self.losses(output_aux16,gt_semantic_seg)
            # losses_p=self.losses(x_extra_p,gt_semantic_seg)
            losses=self.losses(output,gt_semantic_seg)
            # losses_stdc_aux=self.losses(output_aux16,gt_semantic_seg)
            return feats,losses,losses_aux
        else:
            output, predict,output_aux16,feats  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            losses=self.losses(predict,gt_semantic_seg)
            losses_stdc=self.losses(output,gt_semantic_seg)
            losses_stdc_aux=self.losses(output_aux16,gt_semantic_seg)
            return feats,losses,losses_stdc,losses_stdc_aux

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



# import torch
# import math
# import torch.nn as nn
# import torch.nn.functional as F
# import cv2
# import numpy as np
# from mmcv.cnn import ConvModule
# from ..builder import HEADS
# from .cascade_decode_head import BaseCascadeDecodeHead
# from .diff_fusion import FeatureFusionModule
# from .harr import HarrUp
# from other_utils.heatmap import save_image,save_heatmap
# from mmseg.ops import resize
# from mmseg.models.losses.detail_loss import DetailAggregateLoss
# from .pid import Bag
# from .diff_head import DiffHead
# from .diff_point import DiffPoint
# from .spnet import SpNet,get_coordsandfeatures,SparseResNet50
# from .spnetv2 import SpNetV2
# from other_utils.split_tensor import split_tensor,restore_tensor
# from .pid import segmenthead
# from .sdd_stdc_head import ShallowNet
# from .pidnet_single import PIDNet
# # from .pidnet import PIDNet
# # from .pidnet_un import PIDNet
# # from .pidnet_distill import PIDNet
# # from .pidnet_stdc import PIDNet
# from mmseg.models.losses.bce_dice_loss import BCEDiceLoss
# from mmseg.models.losses.detail_loss import DetailAggregateLoss
# from .isdhead import RelationAwareFusion
#
# class OhemCrossEntropy(nn.Module):
#     def __init__(self, ignore_label=255, thres=0.7,
#                  min_kept=100000, weight=None):
#         super(OhemCrossEntropy, self).__init__()
#         self.thresh = thres
#         self.min_kept = max(1, min_kept)
#         self.ignore_label = ignore_label
#         self.criterion = nn.CrossEntropyLoss(
#             weight=weight,
#             ignore_index=ignore_label,
#             reduction='none'
#         )
#
#     def _ce_forward(self, score, target):
#
#         loss = self.criterion(score, target)
#
#         return loss
#
#     def _ohem_forward(self, score, target, **kwargs):
#
#         pred = F.softmax(score, dim=1)
#         pixel_losses = self.criterion(score, target).contiguous().view(-1)
#         mask = target.contiguous().view(-1) != self.ignore_label
#
#         tmp_target = target.clone()
#         tmp_target[tmp_target == self.ignore_label] = 0
#         pred = pred.gather(1, tmp_target.unsqueeze(1))
#         pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
#         min_value = pred[min(self.min_kept, pred.numel() - 1)]
#         threshold = max(min_value, self.thresh)
#
#         pixel_losses = pixel_losses[mask][ind]
#         pixel_losses = pixel_losses[pred < threshold]
#         return pixel_losses.mean()
#
#     def forward(self, score, target,name):
#         loss=dict()
#         if not (isinstance(score, list) or isinstance(score, tuple)):
#             score = [score]
#
#         balance_weights = [0.5, 0.5]
#         sb_weights = 0.5
#         if len(balance_weights) == len(score):
#             functions = [self._ce_forward] * \
#                         (len(balance_weights) - 1) + [self._ohem_forward]
#             return sum([
#                 w * func(x, target)
#                 for (w, x, func) in zip(balance_weights, score, functions)
#             ])
#
#         elif len(score) == 1:
#             loss[name+'ohem_loss']=sb_weights * self._ohem_forward(score[0], target)
#             return loss
#
#         else:
#             raise ValueError("lengths of prediction and target are not identical!")
#
#
# class SegmentationHead(nn.Module):
#     def __init__(self, conv_cfg, norm_cfg, act_cfg, in_channels, mid_channels, n_classes, *args, **kwargs):
#         super(SegmentationHead, self).__init__()
#
#         self.conv_bn_relu = ConvModule(in_channels, mid_channels, 3,
#                                        stride=1,
#                                        padding=1,
#                                        conv_cfg=conv_cfg,
#                                        norm_cfg=norm_cfg,
#                                        act_cfg=act_cfg)
#
#         self.conv_out = nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=True)
#
#     def forward(self, x):
#         x = self.conv_bn_relu(x)
#         x = self.conv_out(x)
#         return x
#
#
# class Lap_Pyramid_Conv(nn.Module):
#     def __init__(self, num_high=3, gauss_chl=3):
#         super(Lap_Pyramid_Conv, self).__init__()
#
#         self.num_high = num_high
#         self.gauss_chl = gauss_chl
#         self.kernel = self.gauss_kernel()
#
#     def gauss_kernel(self, device=torch.device('cuda')):
#         kernel = torch.tensor([[1., 4., 6., 4., 1],
#                                [4., 16., 24., 16., 4.],
#                                [6., 24., 36., 24., 6.],
#                                [4., 16., 24., 16., 4.],
#                                [1., 4., 6., 4., 1.]])
#         kernel /= 256.
#         kernel = kernel.repeat(self.gauss_chl, 1, 1, 1)
#         kernel = kernel.to(device)
#         return kernel
#
#     def downsample(self, x):
#         return x[:, :, ::2, ::2]
#
#     def upsample(self, x):
#         cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
#         cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
#         cc = cc.permute(0, 1, 3, 2)
#         cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
#         cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
#         x_up = cc.permute(0, 1, 3, 2)
#         return self.conv_gauss(x_up, 4 * self.kernel)
#
#     def conv_gauss(self, img, kernel):
#         img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
#         out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
#         return out
#
#     def pyramid_decom(self, img):
#         current = img
#         pyr = []
#         for _ in range(self.num_high):
#             filtered = self.conv_gauss(current, self.kernel)
#             down = self.downsample(filtered)
#             up = self.upsample(down)
#             if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
#                 up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
#             diff = current - up
#             pyr.append(diff)
#             current = down
#         return pyr
#
#
#
#
#
#
# @HEADS.register_module()
# class SingleDiffHead(BaseCascadeDecodeHead):
#     def __init__(self, down_ratio, prev_channels,img_size, reduce=False,decoder_flag=3, **kwargs):
#         super(SingleDiffHead, self).__init__(**kwargs)
#         self.decoder_flag=decoder_flag
#         self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
#         self.shallow_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
#                 type='BCEDiceLoss'))
#         self.down_ratio = down_ratio
#         self.stdc_net = ShallowNet(in_channels=3, pretrain_model="/root/autodl-tmp/pretrained models/STDCNet813M_73.91.tar",num_classes=self.num_classes)
#         self.pid=PIDNet(m=2, n=3, num_classes=self.num_classes, planes=32, ppm_planes=96, head_planes=128, augment=True)
#         h,w=img_size
#         self.ohem=OhemCrossEntropy()
#         self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
#         self.reduce = Reducer() if reduce else None
#         self.conv_seg_aux_16 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
#                                                 self.channels // 2, self.num_classes, kernel_size=1)
#         self.convert_shallow16=nn.Conv2d(512,self.channels,stride=1,kernel_size=3,padding=1)
#         self.fuse8 = RelationAwareFusion(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg, ext=2)
#         self.shallow_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
#                 type='BCEDiceLoss'))
#
#     def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
#         """Forward function."""
#         decoder_flag=self.decoder_flag
#         # prymaid_results = self.lap_prymaid_conv.pyramid_decom(inputs)
#         #
#         # high_residual_1 = prymaid_results[0]
#         # high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
#         #                                 align_corners=False)
#         #
#         # high_residual_input = torch.cat([high_residual_1, high_residual_2], dim=1)
#         shallow_feat8, shallow_feat16=self.stdc_net(inputs)
#         shallow_feat16=self.convert_shallow16(shallow_feat16)
#         _, aux_feat8, fused_feat_8 = self.fuse8(shallow_feat8, shallow_feat16)
#         output = self.cls_seg(fused_feat_8)
#
#
#         predict = self.pid.forward_sp(fused_feat_8,output)
#         feats=[]
#         if decoder_flag==1:
#             if train_flag:
#                 loss_shallow_diff, diff_map, diff_pred_shallow = self.shallow_diff.forward_train_diff(output, img_metas,
#                                                                                                       gt, train_cfg)
#                 diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
#                                            align_corners=self.align_corners)
#                 diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
#                 diff_pred_shallow_sig = diff_pred_shallow
#                 diff_pred_shallow = (diff_pred_shallow > 0.5).float()
#                 gt[diff_pred_shallow == 0] = 255
#                 return output,output_aux16,predict,feats ,loss_shallow_diff
#             else:
#                 return predict
#         elif decoder_flag==2:
#             if train_flag:
#                 return
#             else:
#                 return
#         else:
#             if train_flag:
#                 output_aux16 = self.conv_seg_aux_16(aux_feat8)
#                 return output,output_aux16,predict,feats
#             else:
#                 return predict
#
#
#
#
#
#
#
#
#
#
#
#     def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg,mask=None):
#         if self.decoder_flag==1:
#             output,output_aux16, predict,feats,loss_shallow_diff  = self.forward(
#                 inputs, prev_output,
#                 mask=mask,
#                 gt=gt_semantic_seg,
#                 img_metas=img_metas,
#                 train_cfg=train_cfg,
#                 diff_pred_deep=mask)
#             # predict = F.interpolate(predict, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
#             # x_extra_p = F.interpolate(x_extra_p, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
#             # # losses_boundary = self.detail_loss(x8_cls, gt_semantic_seg.squeeze(1))
#             # losses_p = self.ohem(x_extra_p, gt_semantic_seg.squeeze(1), 'p_')
#             # losses_p['p_ohem_loss']=losses_p['p_ohem_loss']*0.4
#             # losses = self.ohem(predict, gt_semantic_seg.squeeze(1), 'middle_')
#             losses=self.losses(predict,gt_semantic_seg)
#             # losses_p=self.losses(x_extra_p,gt_semantic_seg)
#             losses_stdc=self.losses(output,gt_semantic_seg)
#             losses_stdc_aux=self.losses(output_aux16,gt_semantic_seg)
#
#             return feats,losses,losses_stdc,losses_stdc_aux,loss_shallow_diff
#         elif self.decoder_flag==2:
#             x_extra_p, predict,output,loss_diff1  = self.forward(
#                 inputs, prev_output,
#                 mask=mask,
#                 gt=gt_semantic_seg,
#                 img_metas=img_metas,
#                 train_cfg=train_cfg,
#                 diff_pred_deep=mask)
#             predict = F.interpolate(predict, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
#             x_extra_p = F.interpolate(x_extra_p, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
#             output=F.interpolate(output, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
#             # losses_boundary = self.detail_loss(x8_cls, gt_semantic_seg.squeeze(1))
#             losses_p = self.ohem(x_extra_p, gt_semantic_seg.squeeze(1), 'p_')
#             losses_p['p_ohem_loss']=losses_p['p_ohem_loss']*0.4
#             losses_middle = self.ohem(predict, gt_semantic_seg.squeeze(1), 'middle_')
#             losses_middle['middle_ohem_loss']=losses_middle['middle_ohem_loss']*0.5
#             losses = self.ohem(output, gt_semantic_seg.squeeze(1), 'output_')
#             feats=[]
#             return feats,losses,losses_middle,losses_p,loss_diff1
#             return
#
#         else:
#             output,output_aux16, predict,feats  = self.forward(
#                 inputs, prev_output,
#                 mask=mask,
#                 gt=gt_semantic_seg,
#                 img_metas=img_metas,
#                 train_cfg=train_cfg,
#                 diff_pred_deep=mask)
#             # predict = F.interpolate(predict, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
#             # x_extra_p = F.interpolate(x_extra_p, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
#             # # losses_boundary = self.detail_loss(x8_cls, gt_semantic_seg.squeeze(1))
#             # losses_p = self.ohem(x_extra_p, gt_semantic_seg.squeeze(1), 'p_')
#             # losses_p['p_ohem_loss']=losses_p['p_ohem_loss']*0.4
#             # losses = self.ohem(predict, gt_semantic_seg.squeeze(1), 'middle_')
#             losses=self.losses(predict,gt_semantic_seg)
#             # losses_p=self.losses(x_extra_p,gt_semantic_seg)
#             losses_stdc=self.losses(output,gt_semantic_seg)
#             losses_stdc_aux=self.losses(output_aux16,gt_semantic_seg)
#
#             return feats,losses,losses_stdc,losses_stdc_aux
#
#     def forward_test(self, inputs, prev_output, img_metas, test_cfg,mask=None):
#         """Forward function for testing.
#
#         Args:
#             inputs (list[Tensor]): List of multi-level img features.
#             prev_output (Tensor): The output of previous decode head.
#             img_metas (list[dict]): List of image info dict where each dict
#                 has: 'img_shape', 'scale_factor', 'flip', and may also contain
#                 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#                 For details on the values of these keys see
#                 `mmseg/datasets/pipelines/formatting.py:Collect`.
#             test_cfg (dict): The testing config.
#
#         Returns:
#             Tensor: Output segmentation map.
#         """
#
#         return self.forward(inputs, prev_output, False,diff_pred_deep=mask)
