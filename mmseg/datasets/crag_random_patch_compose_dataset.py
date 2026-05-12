import math
import os
import random

import mmcv
import numpy as np
from torch.utils.data import Dataset

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose


@DATASETS.register_module()
class CragRandomPatchComposeDataset(Dataset):
    """Compose one large Crag sample from random patches of Crag images."""

    CLASSES = ('background', 'gland')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self,
                 data_root,
                 img_dir='train/Images',
                 ann_dir='train/Annotation',
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 patch_size=(510, 510),
                 out_size=(1530, 1530),
                 ignore_index=255,
                 pipeline=None):
        self.data_root = data_root
        self.img_dir = self._join_root(data_root, img_dir)
        self.ann_dir = self._join_root(data_root, ann_dir)
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.patch_h, self.patch_w = patch_size
        self.out_h, self.out_w = out_size
        self.ignore_index = ignore_index
        self.pipeline = Compose(pipeline)

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

    def _pad_to_patch_size(self, img, label):
        h, w = img.shape[:2]
        target_h = max(h, self.patch_h)
        target_w = max(w, self.patch_w)
        if target_h == h and target_w == w:
            return img, label

        padded_img = np.zeros((target_h, target_w, 3), dtype=img.dtype)
        padded_label = np.full(
            (target_h, target_w), self.ignore_index, dtype=label.dtype)
        padded_img[:h, :w] = img
        padded_label[:h, :w] = label
        return padded_img, padded_label

    def _sample_one_patch(self):
        info = random.choice(self.img_infos)
        img, label = self._read_img_label(info)
        img, label = self._pad_to_patch_size(img, label)

        h, w = img.shape[:2]
        y1 = random.randint(0, h - self.patch_h)
        x1 = random.randint(0, w - self.patch_w)
        y2 = y1 + self.patch_h
        x2 = x1 + self.patch_w

        return img[y1:y2, x1:x2].copy(), label[y1:y2, x1:x2].copy()

    def _compose_large_image(self):
        grid_rows = math.ceil(self.out_h / self.patch_h)
        grid_cols = math.ceil(self.out_w / self.patch_w)
        canvas_h = grid_rows * self.patch_h
        canvas_w = grid_cols * self.patch_w

        canvas_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas_label = np.full(
            (canvas_h, canvas_w), self.ignore_index, dtype=np.uint8)

        for r in range(grid_rows):
            for c in range(grid_cols):
                patch_img, patch_label = self._sample_one_patch()
                y1 = r * self.patch_h
                y2 = y1 + self.patch_h
                x1 = c * self.patch_w
                x2 = x1 + self.patch_w
                canvas_img[y1:y2, x1:x2] = patch_img
                canvas_label[y1:y2, x1:x2] = patch_label

        return canvas_img[:self.out_h, :self.out_w], canvas_label[
            :self.out_h, :self.out_w]

    def _pack_results(self, img, gt, filename):
        results = dict(
            img_info=dict(filename=filename),
            ann_info=dict(seg_map=filename),
            filename=filename,
            ori_filename=filename,
            img=img,
            gt_semantic_seg=gt)

        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_fields'] = ['img']
        results['seg_fields'] = ['gt_semantic_seg']

        return self.pipeline(results)

    def __getitem__(self, idx):
        img, gt = self._compose_large_image()
        return self._pack_results(img, gt, f'crag_patch_syn_{idx}.png')
