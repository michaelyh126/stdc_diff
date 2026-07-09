import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from ..builder import SEGMENTORS
from .stdc import Stdc


@SEGMENTORS.register_module()
class StdcSAPR(Stdc):
    """STDC wrapper with Source-Aware Patch Relation training loss.

    It expects ``MyStdcSAPRHead`` as refine head. The original STDC head is not
    modified, so other STDC experiments keep exactly the same behavior.
    """

    def __init__(self,
                 enable_sapr=True,
                 sapr_loss_weight=0.1,
                 sapr_num_tokens=4,
                 sapr_context_radius=1,
                 sapr_temperature=0.07,
                 sapr_grid_shape=(4, 4),
                 sapr_loss_type='qkv',
                 sapr_pool_size=4,
                 sapr_pool_with_max=True,
                 **kwargs):
        super(StdcSAPR, self).__init__(**kwargs)
        self.enable_sapr = enable_sapr
        self.sapr_loss_weight = sapr_loss_weight
        self.sapr_num_tokens = sapr_num_tokens
        self.sapr_context_radius = sapr_context_radius
        self.sapr_temperature = sapr_temperature
        self.sapr_grid_shape = sapr_grid_shape
        self.sapr_loss_type = sapr_loss_type
        self.sapr_pool_size = sapr_pool_size
        self.sapr_pool_with_max = sapr_pool_with_max

        if self.sapr_loss_type not in ('qkv', 'pool'):
            raise ValueError('sapr_loss_type must be "qkv" or "pool", '
                             f'got {self.sapr_loss_type}')

        channels = self.refine_head.channels
        self.sapr_patch_queries = nn.Parameter(
            torch.randn(sapr_num_tokens, channels) * 0.02)
        self.sapr_slot_queries = nn.Parameter(
            torch.randn(sapr_num_tokens, channels) * 0.02)
        self.sapr_q_proj = nn.Linear(channels, channels, bias=False)
        self.sapr_k_proj = nn.Linear(channels, channels, bias=False)
        self.sapr_v_proj = nn.Linear(channels, channels, bias=False)
        self.sapr_out_proj = nn.Linear(channels, channels)

    @staticmethod
    def _flatten_group_images(img):
        if img is None:
            return None, 0, 0
        if img.dim() == 5:
            groups, count, channels, height, width = img.shape
            return img.reshape(groups * count, channels, height,
                               width), groups, count
        if img.dim() == 4:
            return img, img.shape[0], 1
        raise ValueError(f'Unsupported image tensor shape: {tuple(img.shape)}')

    @staticmethod
    def _flatten_group_labels(gt_semantic_seg):
        if gt_semantic_seg.dim() == 5:
            groups, count, channels, height, width = gt_semantic_seg.shape
            return gt_semantic_seg.reshape(groups * count, channels, height,
                                           width)
        return gt_semantic_seg

    @staticmethod
    def _expand_img_metas(img_metas, num_pseudo):
        if num_pseudo == 1:
            return img_metas
        expanded = []
        for meta in img_metas:
            for pseudo_idx in range(num_pseudo):
                pseudo_meta = meta.copy()
                pseudo_meta['sapr_pseudo_index'] = pseudo_idx
                expanded.append(pseudo_meta)
        return expanded

    def _forward_head_with_sapr_feature(self, img, train_flag, img_metas=None):
        if not hasattr(self.refine_head, 'forward_with_feature'):
            raise AttributeError('StdcSAPR requires decode_head type '
                                 'MyStdcSAPRHead.')
        outputs = self.refine_head.forward_with_feature(
            img,
            None,
            train_flag=train_flag,
            img_metas=img_metas,
            train_cfg=self.train_cfg)
        if train_flag:
            output, aux_output, feature = outputs
            return (output, aux_output), feature
        output, feature = outputs
        return output, feature

    def _forward_real_sapr_feature(self, real_img):
        was_training = self.refine_head.training
        self.refine_head.eval()
        with torch.no_grad():
            _, real_feat = self._forward_head_with_sapr_feature(
                real_img, train_flag=False)
        if was_training:
            self.refine_head.train()
        return real_feat.detach()

    @staticmethod
    def _box_to_feature_range(feat, box, image_size):
        _, feat_h, feat_w = feat.shape
        img_h, img_w = image_size
        x1, y1, x2, y2 = [
            float(v.item()) if torch.is_tensor(v) else float(v)
            for v in box
        ]

        fx1 = int(math.floor(x1 * feat_w / img_w))
        fy1 = int(math.floor(y1 * feat_h / img_h))
        fx2 = int(math.ceil(x2 * feat_w / img_w))
        fy2 = int(math.ceil(y2 * feat_h / img_h))

        fx1 = max(0, min(fx1, feat_w - 1))
        fy1 = max(0, min(fy1, feat_h - 1))
        fx2 = max(fx1 + 1, min(fx2, feat_w))
        fy2 = max(fy1 + 1, min(fy2, feat_h))
        return fx1, fy1, fx2, fy2

    @staticmethod
    def _grid_boxes(image_size, grid_shape, device):
        img_h, img_w = image_size
        grid_h, grid_w = grid_shape
        boxes = []
        for row in range(grid_h):
            y1 = round(row * img_h / grid_h)
            y2 = round((row + 1) * img_h / grid_h)
            for col in range(grid_w):
                x1 = round(col * img_w / grid_w)
                x2 = round((col + 1) * img_w / grid_w)
                boxes.append([x1, y1, x2, y2])
        return torch.tensor(boxes, dtype=torch.float32, device=device)

    @staticmethod
    def _box_to_grid_index(box, image_size, grid_shape):
        img_h, img_w = image_size
        grid_h, grid_w = grid_shape
        x1, y1, x2, y2 = [
            float(v.item()) if torch.is_tensor(v) else float(v)
            for v in box
        ]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        col = int(math.floor(cx * grid_w / img_w))
        row = int(math.floor(cy * grid_h / img_h))
        col = max(0, min(col, grid_w - 1))
        row = max(0, min(row, grid_h - 1))
        return row * grid_w + col

    @staticmethod
    def _neighbor_indices(slot_idx, grid_shape, masked_indices, radius):
        grid_h, grid_w = grid_shape
        row = slot_idx // grid_w
        col = slot_idx % grid_w
        neighbors = []
        masked_indices = set(masked_indices)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                nr = row + dr
                nc = col + dc
                if nr < 0 or nr >= grid_h or nc < 0 or nc >= grid_w:
                    continue
                idx = nr * grid_w + nc
                if idx not in masked_indices:
                    neighbors.append(idx)
        return neighbors

    def _attend_tokens(self, queries, values):
        q = self.sapr_q_proj(queries)
        k = self.sapr_k_proj(values)
        v = self.sapr_v_proj(values)
        scale = math.sqrt(q.size(-1))
        attn = torch.matmul(q, k.transpose(0, 1)) / scale
        tokens = torch.matmul(F.softmax(attn, dim=-1), v)
        return self.sapr_out_proj(tokens)

    def _box_tokens(self, feat, box, image_size):
        fx1, fy1, fx2, fy2 = self._box_to_feature_range(
            feat, box, image_size)
        values = feat[:, fy1:fy2, fx1:fx2].flatten(1).transpose(0, 1)
        return self._attend_tokens(self.sapr_patch_queries, values)

    def _grid_box_tokens(self, feat, boxes, image_size):
        image_tokens = []
        for img_idx in range(feat.size(0)):
            patch_tokens = [
                self._box_tokens(feat[img_idx], box, image_size)
                for box in boxes
            ]
            image_tokens.append(torch.stack(patch_tokens, dim=0))
        return torch.stack(image_tokens, dim=0)

    def _slot_tokens(self, context_tokens):
        values = context_tokens.reshape(-1, context_tokens.size(-1))
        return self._attend_tokens(self.sapr_slot_queries, values)

    def _match_logits(self, slot_tokens, candidate_tokens):
        slot_tokens = F.normalize(slot_tokens, dim=-1)
        candidate_tokens = F.normalize(candidate_tokens, dim=-1)
        sim = torch.einsum('qc,nkc->nqk', slot_tokens, candidate_tokens)
        logits = sim.max(dim=2).values.mean(dim=1)
        return logits / self.sapr_temperature

    def _box_pool_tokens(self, feat, box, image_size):
        fx1, fy1, fx2, fy2 = self._box_to_feature_range(
            feat, box, image_size)
        region = feat[:, fy1:fy2, fx1:fx2].unsqueeze(0)
        pool_size = (self.sapr_pool_size, self.sapr_pool_size)
        avg_tokens = F.adaptive_avg_pool2d(region, pool_size)
        avg_tokens = avg_tokens.squeeze(0).flatten(1).transpose(0, 1)
        if not self.sapr_pool_with_max:
            return avg_tokens

        max_tokens = F.adaptive_max_pool2d(region, pool_size)
        max_tokens = max_tokens.squeeze(0).flatten(1).transpose(0, 1)
        return torch.cat([avg_tokens, max_tokens], dim=-1)

    def _grid_box_pool_tokens(self, feat, boxes, image_size):
        image_tokens = []
        for img_idx in range(feat.size(0)):
            patch_tokens = [
                self._box_pool_tokens(feat[img_idx], box, image_size)
                for box in boxes
            ]
            image_tokens.append(torch.stack(patch_tokens, dim=0))
        return torch.stack(image_tokens, dim=0)

    def _pool_match_logits(self, context_tokens, candidate_tokens):
        context_tokens = F.normalize(context_tokens, dim=-1)
        candidate_tokens = F.normalize(candidate_tokens, dim=-1)
        sim = torch.einsum('qc,nkc->nqk', context_tokens, candidate_tokens)
        logits = sim.max(dim=2).values.mean(dim=1)
        return logits / self.sapr_temperature

    def _sapr_pool_loss(self, pseudo_feat, real_feat, sapr_map, num_groups,
                        num_pseudo, num_real, pseudo_size, real_size,
                        grid_shape):
        if sapr_map is None or sapr_map.numel() == 0:
            zero = pseudo_feat.sum() * 0
            return zero, zero.detach()
        if sapr_map.dim() == 2:
            sapr_map = sapr_map.unsqueeze(0)

        sapr_map = sapr_map.to(pseudo_feat.device)
        pseudo_boxes = self._grid_boxes(pseudo_size, grid_shape,
                                        pseudo_feat.device)
        real_boxes = self._grid_boxes(real_size, grid_shape, real_feat.device)
        num_grid = pseudo_boxes.size(0)

        pseudo_tokens = self._grid_box_pool_tokens(pseudo_feat, pseudo_boxes,
                                                   pseudo_size)
        real_tokens = self._grid_box_pool_tokens(real_feat, real_boxes,
                                                 real_size)
        pseudo_tokens = pseudo_tokens.reshape(num_groups, num_pseudo,
                                              num_grid,
                                              self.sapr_pool_size**2, -1)
        real_tokens = real_tokens.reshape(num_groups, num_real, num_grid,
                                          self.sapr_pool_size**2, -1)

        losses = []
        correct = []
        for group_idx in range(num_groups):
            group_pairs = sapr_map[group_idx]
            masked_by_real = {}
            for pair in group_pairs:
                real_idx = int(pair[1].item())
                slot_idx = self._box_to_grid_index(pair[6:10], real_size,
                                                   grid_shape)
                masked_by_real.setdefault(real_idx, set()).add(slot_idx)

            group_candidates = pseudo_tokens[group_idx].reshape(
                num_pseudo * num_grid, self.sapr_pool_size**2, -1)
            for pair in group_pairs:
                pseudo_idx = int(pair[0].item())
                real_idx = int(pair[1].item())
                if pseudo_idx < 0 or real_idx < 0:
                    continue
                if pseudo_idx >= num_pseudo or real_idx >= num_real:
                    continue

                slot_idx = self._box_to_grid_index(pair[6:10], real_size,
                                                   grid_shape)
                target_patch_idx = self._box_to_grid_index(
                    pair[2:6], pseudo_size, grid_shape)
                target_idx = pseudo_idx * num_grid + target_patch_idx

                neighbor_ids = self._neighbor_indices(
                    slot_idx, grid_shape, masked_by_real.get(real_idx, set()),
                    self.sapr_context_radius)
                if len(neighbor_ids) == 0:
                    visible_ids = [
                        idx for idx in range(num_grid)
                        if idx not in masked_by_real.get(real_idx, set())
                    ]
                    neighbor_ids = visible_ids
                if len(neighbor_ids) == 0:
                    continue

                context_tokens = real_tokens[group_idx, real_idx,
                                             neighbor_ids].reshape(
                                                 -1, real_tokens.size(-1))
                logits = self._pool_match_logits(context_tokens,
                                                 group_candidates)
                target = logits.new_tensor([target_idx], dtype=torch.long)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target))
                correct.append((logits.argmax(dim=0) == target_idx).float())

        if len(losses) == 0:
            zero = pseudo_feat.sum() * 0
            return zero, zero.detach()

        loss = torch.stack(losses).mean() * self.sapr_loss_weight
        match_acc = torch.stack(correct).mean() * 100.0
        return loss, match_acc.detach()

    def _select_sapr_loss(self, pseudo_feat, real_feat, sapr_map, num_groups,
                          num_pseudo, num_real, pseudo_size, real_size,
                          grid_shape):
        if self.sapr_loss_type == 'qkv':
            return self._sapr_loss(pseudo_feat, real_feat, sapr_map,
                                   num_groups, num_pseudo, num_real,
                                   pseudo_size, real_size, grid_shape)
        return self._sapr_pool_loss(pseudo_feat, real_feat, sapr_map,
                                    num_groups, num_pseudo, num_real,
                                    pseudo_size, real_size, grid_shape)

    def _sapr_loss(self, pseudo_feat, real_feat, sapr_map, num_groups,
                   num_pseudo, num_real, pseudo_size, real_size, grid_shape):
        if sapr_map is None or sapr_map.numel() == 0:
            zero = pseudo_feat.sum() * 0
            return zero, zero.detach()
        if sapr_map.dim() == 2:
            sapr_map = sapr_map.unsqueeze(0)

        sapr_map = sapr_map.to(pseudo_feat.device)
        pseudo_boxes = self._grid_boxes(pseudo_size, grid_shape,
                                        pseudo_feat.device)
        real_boxes = self._grid_boxes(real_size, grid_shape, real_feat.device)
        num_grid = pseudo_boxes.size(0)

        pseudo_tokens = self._grid_box_tokens(pseudo_feat, pseudo_boxes,
                                              pseudo_size)
        real_tokens = self._grid_box_tokens(real_feat, real_boxes, real_size)
        pseudo_tokens = pseudo_tokens.reshape(num_groups, num_pseudo,
                                              num_grid,
                                              self.sapr_num_tokens, -1)
        real_tokens = real_tokens.reshape(num_groups, num_real, num_grid,
                                          self.sapr_num_tokens, -1)

        losses = []
        correct = []
        for group_idx in range(num_groups):
            group_pairs = sapr_map[group_idx]
            masked_by_real = {}
            for pair in group_pairs:
                real_idx = int(pair[1].item())
                slot_idx = self._box_to_grid_index(pair[6:10], real_size,
                                                   grid_shape)
                masked_by_real.setdefault(real_idx, set()).add(slot_idx)

            group_candidates = pseudo_tokens[group_idx].reshape(
                num_pseudo * num_grid, self.sapr_num_tokens, -1)
            for pair in group_pairs:
                pseudo_idx = int(pair[0].item())
                real_idx = int(pair[1].item())
                if pseudo_idx < 0 or real_idx < 0:
                    continue
                if pseudo_idx >= num_pseudo or real_idx >= num_real:
                    continue

                slot_idx = self._box_to_grid_index(pair[6:10], real_size,
                                                   grid_shape)
                target_patch_idx = self._box_to_grid_index(
                    pair[2:6], pseudo_size, grid_shape)
                target_idx = pseudo_idx * num_grid + target_patch_idx

                neighbor_ids = self._neighbor_indices(
                    slot_idx, grid_shape, masked_by_real.get(real_idx, set()),
                    self.sapr_context_radius)
                if len(neighbor_ids) == 0:
                    visible_ids = [
                        idx for idx in range(num_grid)
                        if idx not in masked_by_real.get(real_idx, set())
                    ]
                    neighbor_ids = visible_ids
                if len(neighbor_ids) == 0:
                    continue

                context_tokens = real_tokens[group_idx, real_idx,
                                             neighbor_ids]
                slot_tokens = self._slot_tokens(context_tokens)
                logits = self._match_logits(slot_tokens, group_candidates)
                target = logits.new_tensor([target_idx], dtype=torch.long)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target))
                correct.append((logits.argmax(dim=0) == target_idx).float())

        if len(losses) == 0:
            zero = pseudo_feat.sum() * 0
            return zero, zero.detach()

        loss = torch.stack(losses).mean() * self.sapr_loss_weight
        match_acc = torch.stack(correct).mean() * 100.0
        return loss, match_acc.detach()

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      sapr_real_img=None,
                      sapr_map=None):
        pseudo_img, num_groups, num_pseudo = self._flatten_group_images(img)
        pseudo_gt = self._flatten_group_labels(gt_semantic_seg)
        pseudo_metas = self._expand_img_metas(img_metas, num_pseudo)

        losses = dict()
        head_outputs, pseudo_feat = self._forward_head_with_sapr_feature(
            pseudo_img,
            train_flag=True,
            img_metas=pseudo_metas)
        output, aux_output = head_outputs

        loss_refine = self.refine_head.losses(output, pseudo_gt)
        losses.update(add_prefix(loss_refine, 'refine'))

        loss_aux = self.refine_head.losses(aux_output, pseudo_gt)
        losses.update(add_prefix(loss_aux, 'aux_1'))

        if (self.enable_sapr and sapr_real_img is not None and sapr_map is not None
                and self.sapr_loss_weight > 0):
            real_img, real_groups, num_real = self._flatten_group_images(
                sapr_real_img)
            if real_groups != num_groups:
                raise ValueError('SAPR real image groups must match pseudo '
                                 f'groups, got {real_groups} and {num_groups}')
            real_feat = self._forward_real_sapr_feature(real_img)
            grid_shape = img_metas[0].get('sapr_grid_shape',
                                          self.sapr_grid_shape)
            loss_sapr, acc_sapr_match = self._select_sapr_loss(
                pseudo_feat,
                real_feat,
                sapr_map,
                num_groups,
                num_pseudo,
                num_real,
                pseudo_img.shape[-2:],
                real_img.shape[-2:],
                grid_shape)
            losses['loss_sapr'] = loss_sapr
            losses['acc_sapr_match'] = acc_sapr_match

        return losses
