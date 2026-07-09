import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight


@LOSSES.register_module()
class EntropyTopKCrossEntropyLoss(nn.Module):
    """Cross entropy on high-entropy pixels selected per image."""

    def __init__(self, topk_ratio=0.3, loss_weight=1.0, class_weight=None):
        super(EntropyTopKCrossEntropyLoss, self).__init__()
        assert 0 < topk_ratio <= 1, 'topk_ratio must be in (0, 1].'
        self.topk_ratio = topk_ratio
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)

    def forward(self, cls_score, label, ignore_index=255, **kwargs):
        if label.dim() == 4:
            label = label.squeeze(1)

        class_weight = None
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)

        pixel_loss = F.cross_entropy(
            cls_score,
            label,
            weight=class_weight,
            reduction='none',
            ignore_index=ignore_index)

        with torch.no_grad():
            prob = F.softmax(cls_score.detach(), dim=1)
            entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1)

        selected_losses = []
        for batch_idx in range(cls_score.size(0)):
            valid_mask = label[batch_idx] != ignore_index
            valid_entropy = entropy[batch_idx][valid_mask]
            if valid_entropy.numel() == 0:
                continue
            topk_num = max(1, int(valid_entropy.numel() * self.topk_ratio))
            _, topk_indices = torch.topk(valid_entropy, topk_num)
            valid_loss = pixel_loss[batch_idx][valid_mask]
            selected_losses.append(valid_loss[topk_indices])

        if len(selected_losses) == 0:
            return cls_score.sum() * 0.0

        return self.loss_weight * torch.cat(selected_losses).mean()
