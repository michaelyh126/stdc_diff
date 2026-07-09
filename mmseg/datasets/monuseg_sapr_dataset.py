import copy
import os
import random

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose


@DATASETS.register_module()
class MoNuSegSAPRDataset(Dataset):
    """MoNuSeg pseudo-large dataset with Source-Aware Patch Relation pairs."""

    CLASSES = ('background', 'nucleus')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self,
                 data_root,
                 img_dir='imgs/train',
                 ann_dir='labels/train',
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 out_size=(1000, 1000),
                 grid_shape=(4, 4),
                 num_pseudo=4,
                 num_real=1,
                 mask_pattern='checkerboard',
                 mask_offset=0,
                 ignore_index=255,
                 img_norm_cfg=None,
                 random_patch_pipeline=None,
                 compensate_random_patch_aug=True,
                 random_patch_aug_scale=None):
        self.data_root = data_root
        self.img_dir = self._join_root(data_root, img_dir)
        self.ann_dir = self._join_root(data_root, ann_dir)
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.out_h, self.out_w = out_size
        self.grid_h, self.grid_w = grid_shape
        self.num_pseudo = num_pseudo
        self.num_real = num_real
        self.mask_pattern = mask_pattern
        self.mask_offset = mask_offset
        self.ignore_index = ignore_index
        self.compensate_random_patch_aug = compensate_random_patch_aug
        self.random_patch_aug_scale = self._resolve_random_patch_aug_scale(
            random_patch_aug_scale)
        random_patch_pipeline = self._compensate_random_patch_pipeline(
            random_patch_pipeline)
        self.random_patch_pipeline = (
            Compose(random_patch_pipeline)
            if random_patch_pipeline is not None else None)

        if img_norm_cfg is None:
            img_norm_cfg = dict(
                mean=[164.258, 114.090, 153.999],
                std=[59.028, 59.278, 47.782],
                to_rgb=True)
        self.mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        self.std = np.array(img_norm_cfg['std'], dtype=np.float32)
        self.to_rgb = img_norm_cfg.get('to_rgb', True)

        self.img_infos = self._load_file_pairs()
        if len(self.img_infos) == 0:
            raise ValueError(f'No files found in {self.img_dir}')

    def __len__(self):
        return len(self.img_infos)

    @staticmethod
    def _join_root(data_root, path):
        if os.path.isabs(path):
            return path
        return os.path.join(data_root, path)

    def _load_file_pairs(self):
        img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(self.img_suffix.lower())
        ])

        infos = []
        for img_name in img_files:
            stem = os.path.splitext(img_name)[0]
            ann_name = stem + self.seg_map_suffix
            img_path = os.path.join(self.img_dir, img_name)
            ann_path = os.path.join(self.ann_dir, ann_name)
            if os.path.exists(ann_path):
                infos.append(
                    dict(
                        filename=img_name,
                        img_path=img_path,
                        ann_path=ann_path))
        return infos

    def _read_img_label(self, info):
        img = mmcv.imread(info['img_path'], flag='color')
        label = mmcv.imread(info['ann_path'], flag='unchanged')
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {info['img_path']}")
        if label is None:
            raise FileNotFoundError(f"Cannot read label: {info['ann_path']}")
        if label.ndim == 3:
            label = label[:, :, 0]
        return img, label

    def _fit_to_out_size(self, img, label):
        if img.shape[:2] == (self.out_h, self.out_w):
            return img, label
        img = mmcv.imresize(img, (self.out_w, self.out_h))
        label = mmcv.imresize(
            label, (self.out_w, self.out_h), interpolation='nearest')
        return img, label

    def _grid_boxes(self):
        ys = np.linspace(0, self.out_h, self.grid_h + 1).round().astype(int)
        xs = np.linspace(0, self.out_w, self.grid_w + 1).round().astype(int)
        boxes = []
        for row in range(self.grid_h):
            for col in range(self.grid_w):
                boxes.append((int(xs[col]), int(ys[row]), int(xs[col + 1]),
                              int(ys[row + 1]), row, col))
        return boxes

    def _masked_boxes(self, boxes):
        if self.mask_pattern == 'checkerboard':
            return [
                box for box in boxes
                if (box[4] + box[5] + self.mask_offset) % 2 == 1
            ]
        if self.mask_pattern == 'inverse_checkerboard':
            return [
                box for box in boxes
                if (box[4] + box[5] + self.mask_offset) % 2 == 0
            ]
        if self.mask_pattern == 'all':
            return boxes
        raise ValueError(f'Unsupported SAPR mask pattern: {self.mask_pattern}')

    def _sapr_insert_area_ratio(self):
        boxes = self._grid_boxes()
        masked_boxes = self._masked_boxes(boxes)
        masked_area = sum((x2 - x1) * (y2 - y1)
                          for x1, y1, x2, y2, _, _ in masked_boxes)
        total_area = self.num_pseudo * self.out_h * self.out_w
        if total_area <= 0:
            return 0.0
        return min((self.num_real * masked_area) / float(total_area), 0.99)

    def _resolve_random_patch_aug_scale(self, random_patch_aug_scale):
        if random_patch_aug_scale is not None:
            return float(random_patch_aug_scale)
        if not self.compensate_random_patch_aug:
            return 1.0

        random_patch_ratio = 1.0 - self._sapr_insert_area_ratio()
        return 1.0 / max(random_patch_ratio, 1e-6)

    def _compensate_random_patch_pipeline(self, pipeline):
        if pipeline is None:
            return None

        pipeline = copy.deepcopy(pipeline)
        if self.random_patch_aug_scale <= 1:
            return pipeline

        for transform in pipeline:
            transform_type = transform.get('type')
            if transform_type in ('RandomFlip', 'RandomRotate'):
                if 'prob' in transform:
                    transform['prob'] = min(
                        float(transform['prob']) *
                        self.random_patch_aug_scale, 1.0)
            elif transform_type == 'PhotoMetricDistortion':
                transform['type'] = 'PhotoMetricDistortionProb'
                transform['prob'] = min(0.5 * self.random_patch_aug_scale,
                                        1.0)
            elif transform_type == 'PhotoMetricDistortionProb':
                transform['prob'] = min(
                    float(transform.get('prob', 0.5)) *
                    self.random_patch_aug_scale, 1.0)
        return pipeline

    def _pad_to_crop_size(self, img, label, crop_h, crop_w):
        h, w = img.shape[:2]
        target_h = max(h, crop_h)
        target_w = max(w, crop_w)
        if (target_h, target_w) == (h, w):
            return img, label

        padded_img = np.zeros((target_h, target_w, 3), dtype=img.dtype)
        padded_label = np.full(
            (target_h, target_w), self.ignore_index, dtype=label.dtype)
        padded_img[:h, :w] = img
        padded_label[:h, :w] = label
        return padded_img, padded_label

    def _sample_random_patch(self, crop_h, crop_w):
        info = random.choice(self.img_infos)
        img, label = self._read_img_label(info)
        img, label = self._pad_to_crop_size(img, label, crop_h, crop_w)
        h, w = img.shape[:2]
        y1 = random.randint(0, h - crop_h)
        x1 = random.randint(0, w - crop_w)
        return (img[y1:y1 + crop_h, x1:x1 + crop_w].copy(),
                label[y1:y1 + crop_h, x1:x1 + crop_w].copy())

    def _fill_random_pseudo(self, pseudo_imgs, pseudo_labels, boxes):
        for pseudo_idx in range(self.num_pseudo):
            for x1, y1, x2, y2, _, _ in boxes:
                patch_img, patch_label = self._sample_random_patch(
                    y2 - y1, x2 - x1)
                pseudo_imgs[pseudo_idx, y1:y2, x1:x2] = patch_img
                pseudo_labels[pseudo_idx, y1:y2, x1:x2] = patch_label

    @staticmethod
    def _pipeline_results(img, label, filename):
        results = dict(
            img_info=dict(filename=filename),
            ann_info=dict(seg_map=filename),
            filename=filename,
            ori_filename=filename,
            img=img,
            gt_semantic_seg=label)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img']
        results['seg_fields'] = ['gt_semantic_seg']
        return results

    def _augment_random_pseudo(self, pseudo_imgs, pseudo_labels, idx):
        if self.random_patch_pipeline is None:
            return

        for pseudo_idx in range(self.num_pseudo):
            filename = f'monuseg_sapr_random_{idx}_{pseudo_idx}.png'
            results = self._pipeline_results(pseudo_imgs[pseudo_idx],
                                             pseudo_labels[pseudo_idx],
                                             filename)
            results = self.random_patch_pipeline(results)
            pseudo_imgs[pseudo_idx] = results['img']
            pseudo_labels[pseudo_idx] = results['gt_semantic_seg']

    def _insert_sapr_patches(self, pseudo_imgs, pseudo_labels, real_imgs,
                             real_labels, boxes, masked_boxes):
        slots = [(pseudo_idx, box) for pseudo_idx in range(self.num_pseudo)
                 for box in boxes]
        random.shuffle(slots)
        if len(slots) < len(masked_boxes) * self.num_real:
            raise ValueError('num_pseudo * grid cells must cover SAPR patches')

        sapr_pairs = []
        slot_idx = 0
        for real_idx in range(self.num_real):
            for real_box in masked_boxes:
                pseudo_idx, pseudo_box = slots[slot_idx]
                slot_idx += 1

                rx1, ry1, rx2, ry2, _, _ = real_box
                px1, py1, px2, py2, _, _ = pseudo_box
                patch_img = real_imgs[real_idx, ry1:ry2, rx1:rx2]
                patch_label = real_labels[real_idx, ry1:ry2, rx1:rx2]

                if patch_img.shape[:2] != (py2 - py1, px2 - px1):
                    patch_img = mmcv.imresize(patch_img,
                                              (px2 - px1, py2 - py1))
                    patch_label = mmcv.imresize(
                        patch_label, (px2 - px1, py2 - py1),
                        interpolation='nearest')

                pseudo_imgs[pseudo_idx, py1:py2, px1:px2] = patch_img
                pseudo_labels[pseudo_idx, py1:py2, px1:px2] = patch_label
                sapr_pairs.append([
                    pseudo_idx, real_idx, px1, py1, px2, py2, rx1, ry1, rx2,
                    ry2
                ])
        return np.array(sapr_pairs, dtype=np.float32)

    def _normalize(self, imgs):
        return np.stack([
            mmcv.imnormalize(img.astype(np.float32), self.mean, self.std,
                             self.to_rgb) for img in imgs
        ],
                        axis=0)

    @staticmethod
    def _to_tensor_img(imgs):
        imgs = np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))
        return torch.from_numpy(imgs)

    @staticmethod
    def _to_tensor_seg(labels):
        labels = labels[:, None, :, :].astype(np.int64)
        return torch.from_numpy(np.ascontiguousarray(labels))

    def _select_real_infos(self, idx):
        infos = [self.img_infos[idx % len(self.img_infos)]]
        for _ in range(1, self.num_real):
            infos.append(random.choice(self.img_infos))
        return infos

    def __getitem__(self, idx):
        boxes = self._grid_boxes()
        masked_boxes = self._masked_boxes(boxes)
        real_infos = self._select_real_infos(idx)

        real_imgs = []
        real_labels = []
        for info in real_infos:
            img, label = self._read_img_label(info)
            img, label = self._fit_to_out_size(img, label)
            real_imgs.append(img)
            real_labels.append(label)
        real_imgs = np.stack(real_imgs, axis=0)
        real_labels = np.stack(real_labels, axis=0)

        pseudo_imgs = np.zeros(
            (self.num_pseudo, self.out_h, self.out_w, 3), dtype=np.uint8)
        pseudo_labels = np.full(
            (self.num_pseudo, self.out_h, self.out_w),
            self.ignore_index,
            dtype=np.uint8)
        self._fill_random_pseudo(pseudo_imgs, pseudo_labels, boxes)
        self._augment_random_pseudo(pseudo_imgs, pseudo_labels, idx)
        sapr_map = self._insert_sapr_patches(pseudo_imgs, pseudo_labels,
                                             real_imgs, real_labels, boxes,
                                             masked_boxes)

        img_tensor = self._to_tensor_img(self._normalize(pseudo_imgs))
        gt_tensor = self._to_tensor_seg(pseudo_labels)
        real_tensor = self._to_tensor_img(self._normalize(real_imgs))
        sapr_tensor = torch.from_numpy(sapr_map)

        filename = f'monuseg_sapr_group_{idx}.png'
        meta = dict(
            filename=filename,
            ori_filename=filename,
            ori_shape=(self.out_h, self.out_w, 3),
            img_shape=(self.out_h, self.out_w, 3),
            pad_shape=(self.out_h, self.out_w, 3),
            scale_factor=1.0,
            flip=False,
            flip_direction=None,
            img_norm_cfg=dict(
                mean=self.mean, std=self.std, to_rgb=self.to_rgb),
            sapr_num_pseudo=self.num_pseudo,
            sapr_num_real=self.num_real,
            sapr_grid_shape=(self.grid_h, self.grid_w),
            sapr_random_patch_aug_scale=self.random_patch_aug_scale,
            sapr_real_filenames=[info['filename'] for info in real_infos])

        return dict(
            img=DC(img_tensor, stack=True),
            gt_semantic_seg=DC(gt_tensor, stack=True),
            sapr_real_img=DC(real_tensor, stack=True),
            sapr_map=DC(sapr_tensor, stack=True, pad_dims=None),
            img_metas=DC(meta, cpu_only=True))
