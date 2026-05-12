import math
import os
import random

import cv2
import mmcv
import numpy as np

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class AerialComposeDataset(CustomDataset):
    """Compose a large aerial training sample from random patches.

    Each returned sample is made by randomly choosing a source image for every
    500x500 tile, cropping one random patch, and placing the patches into a
    2500x2500 canvas. It does not alternate with full original images.
    """

    CLASSES = ('background', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self,
                 data_root,
                 img_dir='imgs/train',
                 ann_dir='labels/train',
                 img_suffix='',
                 seg_map_suffix='',
                 patch_size=(500, 500),
                 out_size=(2500, 2500),
                 ignore_index=255,
                 pipeline=None,
                 **kwargs):
        self.patch_h, self.patch_w = patch_size
        self.out_h, self.out_w = out_size

        super().__init__(
            data_root=data_root,
            img_dir=img_dir,
            ann_dir=ann_dir,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            pipeline=pipeline,
            **kwargs)

    def _read_img_label(self, idx):
        info = self.img_infos[idx]
        img_path = os.path.join(self.img_dir, info['filename'])
        ann_path = os.path.join(self.ann_dir, info['ann']['seg_map'])

        img = mmcv.imread(img_path, flag='color')
        label = mmcv.imread(ann_path, flag='unchanged')

        if img is None:
            raise FileNotFoundError(f'Cannot read image: {img_path}')
        if label is None:
            raise FileNotFoundError(f'Cannot read label: {ann_path}')

        if label.ndim == 3:
            label = label[:, :, 0]

        if label.shape[:2] != img.shape[:2]:
            label = cv2.resize(
                label,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST)

        return img, label

    def _resize_pair(self, img, label, out_h, out_w):
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(
            label, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return img, label

    def _crop_patch(self, img, label):
        h, w = img.shape[:2]
        if h < self.patch_h or w < self.patch_w:
            img, label = self._resize_pair(
                img, label, max(h, self.patch_h), max(w, self.patch_w))
            h, w = img.shape[:2]

        y1 = random.randint(0, h - self.patch_h)
        x1 = random.randint(0, w - self.patch_w)
        y2 = y1 + self.patch_h
        x2 = x1 + self.patch_w

        return img[y1:y2, x1:x2], label[y1:y2, x1:x2]

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
                src_idx = random.randint(0, len(self.img_infos) - 1)
                img, label = self._read_img_label(src_idx)
                patch_img, patch_label = self._crop_patch(img, label)

                y1 = r * self.patch_h
                y2 = y1 + self.patch_h
                x1 = c * self.patch_w
                x2 = x1 + self.patch_w

                canvas_img[y1:y2, x1:x2] = patch_img
                canvas_label[y1:y2, x1:x2] = patch_label

        return canvas_img[:self.out_h, :self.out_w], canvas_label[:self.out_h,
                                                                 :self.out_w]

    def __getitem__(self, idx):
        img, gt = self._compose_large_image()
        fake_name = f'aerial_compose_{idx}.png'

        results = dict(
            img_info=dict(filename=fake_name),
            ann_info=dict(seg_map=fake_name),
            filename=fake_name,
            ori_filename=fake_name,
            img=img,
            gt_semantic_seg=gt)

        self.pre_pipeline(results)

        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['flip'] = False
        results['flip_direction'] = None
        results['img_fields'] = ['img']
        results['seg_fields'] = ['gt_semantic_seg']

        return self.pipeline(results)
