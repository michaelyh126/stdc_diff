import torch
import torch.nn as nn
import torch.nn.functional as F
from other_utils.heatmap import save_image,save_heatmap
from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss

@LOSSES.register_module()
class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target,weight=None,ignore_index=255):
        # Binary Cross-Entropy Loss
        pred=pred.squeeze(1)
        mask = (target != ignore_index)
        pred = pred[mask]
        target = target[mask]
        pred=torch.sigmoid(pred)



        # pred_save = (pred > 0.5).float()
        # save_image(pred_save[1].detach().cpu().numpy(), 'new_diff_pred', 'D:\isd\ISDNet-local\diff_dir\pred')
        # save_image(target[1].detach().cpu().numpy(), 'diff_gt', 'D:\isd\ISDNet-local\diff_dir\gt')
        # save_image(pred_save[0].detach().cpu().numpy(), 'new_diff_pred', 'D:\isd\ISDNet-local\diff_dir\pred')
        # save_image(target[0].detach().cpu().numpy(), 'diff_gt', 'D:\isd\ISDNet-local\diff_dir\gt')

        target = target.float()
        bce_loss = self.bce_loss(pred, target)

        # Dice Loss
        intersection = (pred * target).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        # Combined Loss
        combined_loss = self.alpha * bce_loss + self.beta * dice_loss

        return combined_loss
