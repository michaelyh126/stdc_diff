import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss


def focal_loss(pred,
               target,
               weight=None,
               gamma=2.0,
               class_weight=None,
               reduction='mean',
               avg_factor=None,
               ignore_index=255):
    """Softmax focal loss for semantic segmentation.

    Args:
        pred (Tensor): Logits with shape [N, C] or [N, C, H, W].
        target (Tensor): Labels with shape [N] or [N, H, W].
        weight (Tensor, optional): Element-wise weight.
        gamma (float): Focusing parameter in focal loss.
        class_weight (Tensor, optional): Per-class weight.
        reduction (str): Reduction method, options are 'none', 'mean', 'sum'.
        avg_factor (float, optional): Average factor for reduction.
        ignore_index (int): Label index to ignore.
    """
    assert pred.dim() in (2, 4), \
        'Only pred shape [N, C] or [N, C, H, W] is supported'
    assert target.dim() == pred.dim() - 1, \
        'Target shape must match prediction shape without channel dimension'

    log_prob = F.log_softmax(pred, dim=1)
    prob = log_prob.exp()

    valid_mask = (target >= 0) & (target != ignore_index)
    safe_target = target.clone()
    safe_target[~valid_mask] = 0

    gather_index = safe_target.unsqueeze(1)
    log_pt = log_prob.gather(1, gather_index).squeeze(1)
    pt = prob.gather(1, gather_index).squeeze(1)

    loss = -((1 - pt).pow(gamma)) * log_pt

    if class_weight is not None:
        loss = loss * class_weight[safe_target]

    loss = loss * valid_mask.float()

    if weight is not None:
        weight = weight.float() * valid_mask.float()
    else:
        weight = valid_mask.float()

    if avg_factor is None and reduction == 'mean':
        avg_factor = weight.sum().clamp_min(1.0)

    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):
    """Focal loss used in GPWFormer.

    The paper states that focal loss with gamma=2 is used during training.
    """

    def __init__(self,
                 gamma=2.0,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_cls = self.loss_weight * focal_loss(
            cls_score,
            label,
            weight=weight,
            gamma=self.gamma,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index)
        return loss_cls
