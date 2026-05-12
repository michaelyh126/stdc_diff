import math
import os
import random

import mmcv
import numpy as np

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class DeepGlobeRandomPatchComposeDataset(CustomDataset):
    """Compose one DeepGlobe training sample from random patch images."""

    CLASSES = ('unknown', 'urban', 'agriculture', 'rangeland', 'forest',
               'water', 'barren')
    PALETTE = [[0, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255],
               [0, 255, 0], [0, 0, 255], [255, 255, 255]]

    def __init__(self,
                 data_root,
                 img_dir='img_dir/train',
                 ann_dir='rgb2id/train',
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 patch_size=(306, 306),
                 out_size=(1224, 1224),
                 ignore_index=255,
                 pipeline=None,
                 **kwargs):
        self.patch_h, self.patch_w = patch_size
        self.out_h, self.out_w = out_size
        self.has_random_flip = self._pipeline_has_random_flip(pipeline)

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

        if len(self.img_infos) == 0:
            raise ValueError(f'No files found in {self.img_dir}')

    @staticmethod
    def _pipeline_has_random_flip(pipeline):
        if pipeline is None:
            return False
        return any(t.get('type') == 'RandomFlip' for t in pipeline
                   if isinstance(t, dict))

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
        src_idx = random.randint(0, len(self.img_infos) - 1)
        img, label = self._read_img_label(src_idx)
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

        return (canvas_img[:self.out_h, :self.out_w],
                canvas_label[:self.out_h, :self.out_w])

    def __getitem__(self, idx):
        img, gt = self._compose_large_image()
        fake_name = f'deepglobe_patch_compose_{idx}.png'

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
        if not self.has_random_flip:
            results['flip'] = False
            results['flip_direction'] = None
        results['img_fields'] = ['img']
        results['seg_fields'] = ['gt_semantic_seg']

        return self.pipeline(results)
