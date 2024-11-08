from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy
from other_utils.heatmap import save_image,save_heatmap


class RefineBaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels=128,
                 channels=128,
                 *,
                 num_classes,
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
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(RefineBaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    # 获得语义分支的结果减去ground_truth，即预测错误的点
    def get_uncertainty_map_gt(self,predict_img,ground_truth_img):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predict_img=torch.argmax(predict_img,dim=1).unsqueeze(1)
        mask = ground_truth_img != 255
        # diff=predict_img[mask]-ground_truth_img[mask]
        diff = torch.where(mask, predict_img - ground_truth_img, torch.tensor(255).to(device)).squeeze(1)
        # diff = predict_img - ground_truth_img.squeeze(1)

        # if predict_img.grad_fn==None:
        #     from other_utils.heatmap import save_heatmap,save_image
        #     save_heatmap(diff.cpu().numpy(),filename='diff',channel=0)

        diff = torch.abs(diff)
        diff[(diff != 0) & (diff != 255)] = 1
        # diff[diff != 0] = 1
        return diff

    def sp_loss(self,inputs,mask, img_metas, gt_semantic_seg, train_cfg):
        mask0 = torch.sum(mask == 1)
        mask = mask.to(torch.int64)
        mask1=torch.sum(mask==1)
        gt_semantic_seg[mask == 0] = 255
        inputs_cls=torch.argmax(inputs,dim=1)
        count_of_ones = torch.sum(inputs_cls == 1)
        gt_count_of_ones = torch.sum(gt_semantic_seg == 1)
        gt_count_of_zeros = torch.sum(gt_semantic_seg == 0)
        point_num=gt_count_of_ones+gt_count_of_zeros
        losses = self.losses(inputs, gt_semantic_seg)

        return losses



    def forward_train_diff(self, inputs, img_metas=None, gt_semantic_seg=None, train_cfg=None):
        seg_logit=inputs
        top_two_values, _ = torch.topk(seg_logit, k=2, dim=1)
        abs_diff = torch.abs(top_two_values[:, 0, :, :] - top_two_values[:, 1, :, :])
        # min_value = abs_diff.min()
        # max_value = abs_diff.max()
        # inverted_diff = max_value - abs_diff + min_value
        diff_map,diff_pred=self.forward(abs_diff.unsqueeze(1))
        if gt_semantic_seg!=None:
            seg_logit=resize(seg_logit,size=gt_semantic_seg.size()[2:],mode='bilinear',align_corners=self.align_corners)
            diff_gt=self.get_uncertainty_map_gt(seg_logit,gt_semantic_seg)

            save_image(diff_gt[0].squeeze().detach().cpu().numpy(), filename='diff_gt',
                       save_dir='/root/autodl-tmp/isdnet_harr/diff_dir', )
            seg_logit_cls=torch.argmax(seg_logit,dim=1)
            save_image(seg_logit_cls[0].detach().cpu().numpy(), 'deep_pred', '/root/autodl-tmp/isdnet_harr/diff_dir')
            diff_pred_save = resize(input=diff_pred, size=gt_semantic_seg.shape[2:], mode='bilinear',align_corners=self.align_corners)
            diff_pred_save = torch.sigmoid(diff_pred_save)
            diff_pred_save = (diff_pred_save > 0.5).float()
            save_image(diff_pred_save[0].squeeze().detach().cpu().numpy(), filename='diff_pred',
                       save_dir='/root/autodl-tmp/isdnet_harr/diff_dir', )
            save_image(diff_pred_save[1].squeeze().detach().cpu().numpy(), filename='diff_pred1',
                       save_dir='/root/autodl-tmp/isdnet_harr/diff_dir', )

            diff_gt = diff_gt.unsqueeze(1)

            losses = self.losses(diff_pred, diff_gt)
            losses['loss_seg']=0.05*losses['loss_seg']
            # diff_pred = torch.argmax(diff_pred, dim=1)
            # diff_pred = diff_pred.unsqueeze(1)
            return losses,diff_map,diff_pred
        else:
            return diff_pred

    def forward_test_diff(self,inputs, img_metas, test_cfg=None):
        seg_logit = inputs
        top_two_values, _ = torch.topk(seg_logit, k=2, dim=1)
        abs_diff = torch.abs(top_two_values[:, 0, :, :] - top_two_values[:, 1, :, :])

        # min_value = abs_diff.min()
        # max_value = abs_diff.max()
        # inverted_diff = max_value - abs_diff + min_value
        diff_map, diff_pred = self.forward(abs_diff.unsqueeze(1))
        # diff_pred = torch.argmax(diff_pred, dim=1)
        # diff_pred = diff_pred.unsqueeze(1)

        # seg_logit_cls=torch.argmax(seg_logit,dim=1)
        # save_image(seg_logit_cls[0].detach().cpu().numpy(), 'deep_pred', '/root/autodl-tmp/testimage')
        # diff_pred_resize = resize(input=diff_pred, size=(5000,5000), mode='bilinear',align_corners=self.align_corners)
        # save_heatmap(diff_pred_resize[0].detach().cpu().numpy(),filename='diff_heatmap0',save_dir='/root/autodl-tmp/testimage',channel=0)
        return diff_pred


    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # 因为要使用decoder的中间特征图，所以这里forward的的函数是要返回，或者是一个list
        seg_logits, fm_decoder = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses, fm_decoder

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
