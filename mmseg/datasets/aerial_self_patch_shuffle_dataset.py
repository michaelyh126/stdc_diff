import math
import os
import random

import cv2
import mmcv
import numpy as np

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class AerialSelfPatchShuffleDataset(CustomDataset):
    """Compose one aerial sample by shuffling patches from the same image.

    A source image is cropped to ``out_size``, split into ``patch_size`` tiles,
    and reassembled after shuffling the tile order. The image and segmentation
    tiles always use the same shuffle order.
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
        self.grid_rows = math.ceil(self.out_h / self.patch_h)
        self.grid_cols = math.ceil(self.out_w / self.patch_w)
        self.canvas_h = self.grid_rows * self.patch_h
        self.canvas_w = self.grid_cols * self.patch_w
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

        if label.shape[:2] != img.shape[:2]:
            label = cv2.resize(
                label,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST)

        return img, label

    def _crop_to_canvas(self, img, label):
        h, w = img.shape[:2]
        if h < self.canvas_h or w < self.canvas_w:
            raise ValueError(
                f'Image is smaller than target crop size '
                f'{self.canvas_h}x{self.canvas_w}: got {h}x{w}')

        y1 = random.randint(0, h - self.canvas_h)
        x1 = random.randint(0, w - self.canvas_w)
        y2 = y1 + self.canvas_h
        x2 = x1 + self.canvas_w
        return img[y1:y2, x1:x2], label[y1:y2, x1:x2]

    def _shuffle_patches(self, img, label):
        tile_count = self.grid_rows * self.grid_cols
        order = list(range(tile_count))
        random.shuffle(order)
        if tile_count > 1 and order == list(range(tile_count)):
            order[0], order[1] = order[1], order[0]

        canvas_img = np.zeros_like(img)
        canvas_label = np.full_like(label, self.ignore_index)

        for dst_idx, src_idx in enumerate(order):
            src_row = src_idx // self.grid_cols
            src_col = src_idx % self.grid_cols
            dst_row = dst_idx // self.grid_cols
            dst_col = dst_idx % self.grid_cols

            src_y1 = src_row * self.patch_h
            src_y2 = src_y1 + self.patch_h
            src_x1 = src_col * self.patch_w
            src_x2 = src_x1 + self.patch_w
            dst_y1 = dst_row * self.patch_h
            dst_y2 = dst_y1 + self.patch_h
            dst_x1 = dst_col * self.patch_w
            dst_x2 = dst_x1 + self.patch_w

            canvas_img[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2,
                                                           src_x1:src_x2]
            canvas_label[dst_y1:dst_y2, dst_x1:dst_x2] = label[src_y1:src_y2,
                                                               src_x1:src_x2]

        return (canvas_img[:self.out_h, :self.out_w],
                canvas_label[:self.out_h, :self.out_w])

    def _compose_large_image(self, idx):
        img, label = self._read_img_label(idx)
        img, label = self._crop_to_canvas(img, label)
        return self._shuffle_patches(img, label)

    def __getitem__(self, idx):
        src_idx = idx % len(self.img_infos)
        img, gt = self._compose_large_image(src_idx)
        source_name = self.img_infos[src_idx]['filename']
        fake_name = f'aerial_self_patch_shuffle_{idx}_{source_name}'

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
