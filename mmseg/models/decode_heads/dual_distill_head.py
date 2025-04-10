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
from .pid import Bag
from .diff_head import DiffHead
from .diff_point import DiffPoint
from .spnet import SpNet,get_coordsandfeatures,SparseResNet50
from .spnetv2 import SpNetV2
from other_utils.split_tensor import split_tensor,restore_tensor
from .pid import segmenthead
# from .pidnet import PIDNet
from .pidnet_convnext import PIDNet
# from .pidnet_un import PIDNet
# from .pidnet_distill import PIDNet
# from .pidnet_stdc import PIDNet
from mmseg.models.losses.bce_dice_loss import BCEDiceLoss
from mmseg.models.losses.detail_loss import DetailAggregateLoss

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

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DiffDecoder(nn.Module):
    def __init__(self,high_channel,middle_channel,out_channel,num_classes):
        super(Decoder, self).__init__()
        self.high_channel = high_channel
        self.middle_channel = middle_channel
        self.out_channel = out_channel
        self.conv1 = BasicConv2d(high_channel,self.middle_channel,3,1,1)
        self.conv2 = BasicConv2d(self.middle_channel,self.out_channel,3,1,1)
        # self.conv3= BasicConv2d(self.middle_channel,self.out_channel,3,1,1)
        self.seg_out = nn.Conv2d(self.out_channel, num_classes, 1)

    def forward(self,high_feature,diff):
        diff = torch.sigmoid(diff)
        high_feature = diff * high_feature + high_feature
        high_feature=self.conv1(high_feature)
        high_feature=self.conv2(high_feature)
        # high_feature=self.conv3(high_feature)
        predict=self.seg_out(high_feature)
        high_feature = F.interpolate(high_feature, scale_factor=2, mode='bilinear', align_corners=False)
        predict = F.interpolate(predict, scale_factor=2, mode='bilinear', align_corners=False)

        return high_feature,predict

class Decoder(nn.Module):
    def __init__(self,high_channel,middle_channel,out_channel,num_classes):
        super(Decoder, self).__init__()
        self.high_channel = high_channel
        self.middle_channel = middle_channel
        self.out_channel = out_channel
        self.conv1 = BasicConv2d(high_channel,self.middle_channel,3,1,1)
        self.conv2 = BasicConv2d(self.middle_channel,self.middle_channel,3,1,1)
        self.conv3= BasicConv2d(self.middle_channel,self.out_channel,3,1,1)


    def forward(self,high_feature,diff):
        diff = torch.sigmoid(diff)
        high_feature = diff * high_feature
        high_feature=self.conv1(high_feature)
        high_feature=self.conv2(high_feature)
        high_feature=self.conv3(high_feature)
        return high_feature




@HEADS.register_module()
class DualDistillHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels,img_size, reduce=False,decoder_flag=3, **kwargs):
        super(DualDistillHead, self).__init__(**kwargs)
        self.decoder_flag=decoder_flag
        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
        # self.diff1 = DiffHead(in_channels=1, in_index=3, channels=32, dropout_ratio=0.1,
        #                              num_classes=self.num_classes, align_corners=False, loss_decode=dict(
        #         type='BCEDiceLoss'))
        # self.shallow_diff=DiffHead(in_channels=1,in_index=3,channels=64,dropout_ratio=0.1,num_classes=self.num_classes,align_corners=False,loss_decode=dict(
        #         type='BCEDiceLoss'))
        self.down_ratio = down_ratio
        self.pid=PIDNet(m=2, n=3, num_classes=self.num_classes, planes=32, ppm_planes=96, head_planes=128, augment=True)
        h,w=img_size
        # self.spnet1 = SpNet((math.ceil(h/8),math.ceil(w/8)),in_channels=128,out_channels=128, num_classes=self.num_classes)
        # self.spnet2 = SpNet((math.ceil(h/4),math.ceil(w/4)),in_channels=128,out_channels=128, num_classes=self.num_classes)
        # self.spnet3 = SpNet((math.ceil(h/2),math.ceil(w/2)),in_channels=128,out_channels=32, num_classes=self.num_classes)
        # self.spnet4 = SpNet((math.ceil(h),math.ceil(w)),in_channels=32,out_channels=self.num_classes, num_classes=self.num_classes)
        self.detail_loss=DetailAggregateLoss()
        self.ohem=OhemCrossEntropy()
        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
        # self.decoder1=Decoder(128,128,128,self.num_classes)
        # self.decoder2=Decoder(128,128,64,self.num_classes)
        # self.decoder3=Decoder(64,64,32,self.num_classes)
        self.conv_seg_p = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, 64,
                                         32, self.num_classes, kernel_size=1)
        # self.seg=segmenthead(128,128,self.num_classes)
        self.reduce = Reducer() if reduce else None
        self.diff_flag=False
        # self.bag=Bag(512,128)


    def forward(self, inputs, prev_output,  train_flag=True, mask=None,gt=None,img_metas=None,train_cfg=None,diff_pred_deep=None):
        """Forward function."""
        decoder_flag=self.decoder_flag
        prymaid_results = self.lap_prymaid_conv.pyramid_decom(inputs)

        high_residual_1 = prymaid_results[0]
        high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
                                        align_corners=False)
        high_residual_input = torch.cat([high_residual_1, high_residual_2], dim=1)
        x_extra_p, predict,fusion = self.pid.forward(high_residual_input)
        output=predict
        feats=[]
        if decoder_flag==1:
            if train_flag:
                loss_shallow_diff, diff_map, diff_pred_shallow = self.shallow_diff.forward_train_diff(output, img_metas,
                                                                                                      gt,
                                                                                                      train_cfg)
                diff_pred_shallow_up = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
                                   align_corners=self.align_corners)
                diff_pred_shallow_up = torch.sigmoid(diff_pred_shallow_up)
                diff_pred_shallow_up_sig=diff_pred_shallow_up
                diff_pred_shallow_up = (diff_pred_shallow_up > 0.5).float()
                diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
                diff_pred_shallow_sig=diff_pred_shallow
                diff_pred_shallow = (diff_pred_shallow > 0.5).float()
                output = resize(output, size=inputs.size()[2:], mode='bilinear',
                                   align_corners=self.align_corners)
                # diff_input=torch.concat([inputs,output],dim=1)
                # diff_count_of_ones = torch.sum(diff_pred_shallow == 1)
                diff_input=fusion*diff_pred_shallow
                b, c, _,_ = diff_input.shape
                _,_,h,w=inputs.shape
                # coords, features=get_coordsandfeatures(diff_input)
                # diff_output = self.spnet1(features, coords, b)
                # diff_output=diff_output.dense()
                diff_input = resize(diff_input,size=(math.ceil(h/4),math.ceil(w/4)), mode='bilinear',
                                   align_corners=self.align_corners)
                coords, features=get_coordsandfeatures(diff_input)
                diff_output = self.spnet2(features, coords, b)
                diff_output=diff_output.dense()
                diff_output = resize(diff_output,size=(math.ceil(h/2),math.ceil(w/2)), mode='bilinear',
                                   align_corners=self.align_corners)
                coords, features=get_coordsandfeatures(diff_output)
                diff_output = self.spnet3(features, coords, b)
                diff_output=diff_output.dense()
                diff_output = resize(diff_output,size=(h,w), mode='bilinear',
                                   align_corners=self.align_corners)
                coords, features=get_coordsandfeatures(diff_output)
                diff_output = self.spnet4(features, coords, b)
                diff_output=diff_output.dense()
                # diff_concat=torch.concat([diff_output,output],dim=1)
                diff_fuse=diff_output
                # diff_fuse = self.sp_fuse(diff_concat)
                # sp_loss = self.spnet1.sp_loss(diff_fuse, diff_pred_shallow_up,img_metas,gt, train_cfg)
                mask = (diff_output != 0)
                output[mask] = diff_fuse[mask]
                last_output=output
                # return output, output_aux16, loss_shallow_diff,sp_loss,last_output
                return x_extra_p, predict,loss_shallow_diff,output
            else:
                diff_pred_shallow = self.shallow_diff.forward_test_diff(output, img_metas)
                # diff_pred_shallow = resize(diff_pred_shallow, size=inputs.size()[2:], mode='bilinear',
                #                            align_corners=self.align_corners)
                # output = resize(output, size=inputs.size()[2:], mode='bilinear',
                #                 align_corners=self.align_corners)
                diff_pred_left, diff_pred_top_right, diff_pred_bottom_left, diff_pred_bottom_right = split_tensor(
                    diff_pred_shallow)
                input_top_left, input_top_right, input_bottom_left, input_bottom_right = split_tensor(inputs)
                output_top_left, output_top_right, output_bottom_left, output_bottom_right = split_tensor(output)
                fusion_top_left, fusion_top_right, fusion_bottom_left, fusion_bottom_right = split_tensor(fusion)
                fusion_list=[fusion_top_left, fusion_top_right, fusion_bottom_left, fusion_bottom_right]
                diff_pred_list = [diff_pred_left, diff_pred_top_right, diff_pred_bottom_left, diff_pred_bottom_right]
                inputs_list = [input_top_left, input_top_right, input_bottom_left, input_bottom_right]
                output_list = [output_top_left, output_top_right, output_bottom_left, output_bottom_right]
                last_output_list = []
                for i in range(4):
                    fusion=fusion_list[i]
                    diff_pred_shallow = diff_pred_list[i]
                    inputs = inputs_list[i]
                    output = output_list[i]
                    output = resize(output, size=inputs.size()[2:], mode='bilinear',
                                    align_corners=self.align_corners)
                    diff_pred_shallow = torch.sigmoid(diff_pred_shallow)
                    diff_pred_shallow_sig = diff_pred_shallow
                    diff_pred_shallow = (diff_pred_shallow > 0.5).float()
                    # diff_input = torch.concat([inputs, output], dim=1)
                    diff_input = fusion * diff_pred_shallow
                    b, c, _, _ = diff_input.shape
                    _, _, h, w = inputs.shape
                    diff_input = resize(diff_input, size=(math.ceil(h / 4), math.ceil(w / 4)), mode='bilinear',
                                        align_corners=self.align_corners)
                    coords, features = get_coordsandfeatures(diff_input)
                    n, c = coords.shape
                    if n > 0:
                        diff_output = self.spnet2(features, coords, b)
                        diff_output = diff_output.dense()
                        diff_output = resize(diff_output, size=(math.ceil(h / 2), math.ceil(w / 2)), mode='bilinear',
                                             align_corners=self.align_corners)
                        coords, features = get_coordsandfeatures(diff_output)
                        diff_output = self.spnet3(features, coords, b)
                        diff_output = diff_output.dense()
                        diff_output = resize(diff_output, size=(h, w), mode='bilinear',
                                             align_corners=self.align_corners)
                        coords, features = get_coordsandfeatures(diff_output)
                        diff_output = self.spnet4(features, coords, b)
                        diff_output = diff_output.dense()
                        diff_fuse = diff_output
                        mask = (diff_output != 0)
                        output[mask] = diff_fuse[mask]

                    last_output_list.append(output)
                last_output = restore_tensor(last_output_list[0], last_output_list[1], last_output_list[2],
                                             last_output_list[3])

                return last_output
        elif decoder_flag==2:
            if train_flag:
                loss_diff1, _, diff_pred1 = self.diff1.forward_train_diff(predict, img_metas, gt, train_cfg)
                feats = self.decoder1(fusion, diff_pred1)
                output = self.seg(feats)
                return x_extra_p, predict,output,loss_diff1
            else:
                diff_pred1 = self.diff1.forward_test_diff(predict)
                feats = self.decoder1(fusion, diff_pred1)
                output = self.seg(feats)
                return output
        else:
            if train_flag:
                # loss_shallow_diff, diff_map, diff_pred_shallow = self.shallow_diff.forward_train_diff(output, img_metas,
                #                                                                                       gt,
                #                                                                                       train_cfg)
                return x_extra_p, predict,feats
            else:
                return predict











    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg,mask=None):
        if self.decoder_flag==1:
            gt_ori=gt_semantic_seg.clone()
            x_extra_p, predict,loss_shallow_diff,output  = self.forward(
                inputs, prev_output,
                mask=mask,
                gt=gt_semantic_seg,
                img_metas=img_metas,
                train_cfg=train_cfg,
                diff_pred_deep=mask)
            # predict = F.interpolate(predict, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            # x_extra_p = F.interpolate(x_extra_p, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            # output=F.interpolate(output, gt_semantic_seg.size()[2:], mode='bilinear', align_corners=False)
            # # losses_boundary = self.detail_loss(x8_cls, gt_semantic_seg.squeeze(1))
            # losses_p = self.ohem(x_extra_p, gt_semantic_seg.squeeze(1), 'p_')
            # losses_middle = self.ohem(predict, gt_semantic_seg.squeeze(1), 'middle_')
            # losses = self.ohem(output, gt_semantic_seg.squeeze(1), 'output')
            losses=self.losses(predict,gt_ori)
            losses_p=self.losses(x_extra_p,gt_ori)
            losses_sp=self.losses(output,gt_ori)
            feats=[]
            return feats,losses,losses_p,loss_shallow_diff,losses_sp
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

        else:
            x_extra_p, predict,feats  = self.forward(
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
            losses_p=self.losses(x_extra_p,gt_semantic_seg)

            return feats,losses, losses_p

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
