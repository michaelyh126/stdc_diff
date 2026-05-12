import os
import math
import random
import cv2
import mmcv
import numpy as np

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class Crag510ComposeDataset(CustomDataset):
    CLASSES = ('background', 'gland')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self,
                 data_root,
                 img_dir='train/Images',
                 ann_dir='train/Annotation',
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 patch_size=(510, 510),
                 out_size=(1512, 1516),
                 ignore_index=255,
                 pipeline=None,
                 **kwargs):
        self.patch_h, self.patch_w = patch_size
        self.out_h, self.out_w = out_size
        self.ignore_index = ignore_index

        super().__init__(
            data_root=data_root,
            img_dir=img_dir,
            ann_dir=ann_dir,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            pipeline=pipeline,
            **kwargs
        )

    def _read_img_label(self, idx):
        info = self.img_infos[idx]

        img_path = os.path.join(self.img_dir, info['filename'])
        ann_path = os.path.join(self.ann_dir, info['ann']['seg_map'])

        img = mmcv.imread(img_path, flag='color')
        label = mmcv.imread(ann_path, flag='unchanged')

        if label is None:
            raise FileNotFoundError(f'Cannot read label: {ann_path}')
        if img is None:
            raise FileNotFoundError(f'Cannot read image: {img_path}')

        if label.ndim == 3:
            label = label[:, :, 0]

        return img, label

    def _remove_patch_padding(self, img, label):
        """
        只对 Crag510 patch 去 padding。
        label == 255 代表 padding 区域。
        """
        valid = (label != self.ignore_index)
        if valid.sum() == 0:
            return None, None

        ys, xs = np.where(valid)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1

        img = img[y1:y2, x1:x2]
        label = label[y1:y2, x1:x2]
        return img, label

    def _resize_pair(self, img, label, out_h, out_w):
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return img, label

    def _sample_one_patch(self):
        """
        从 Crag510 中随机采一个 patch：
        1. 读取 510 patch
        2. 去掉 label==255 的 padding
        3. resize 回 510x510
        """
        for _ in range(100):
            idx = random.randint(0, len(self.img_infos) - 1)
            img, label = self._read_img_label(idx)

            # img, label = self._remove_patch_padding(img, label)
            if img is None:
                continue

            img, label = self._resize_pair(img, label, self.patch_h, self.patch_w)
            return img, label

        raise RuntimeError('Failed to sample valid patch from Crag510 after 100 tries.')

    def _compose_large_image(self):
        """
        用多个 510x510 patch 拼成一张大图，再输出固定 1516x1516。
        """
        grid_rows = math.ceil(self.out_h / self.patch_h)
        grid_cols = math.ceil(self.out_w / self.patch_w)

        canvas_h = grid_rows * self.patch_h
        canvas_w = grid_cols * self.patch_w

        canvas_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas_label = np.full((canvas_h, canvas_w), self.ignore_index, dtype=np.uint8)

        for r in range(grid_rows):
            for c in range(grid_cols):
                patch_img, patch_label = self._sample_one_patch()

                y1 = r * self.patch_h
                y2 = y1 + self.patch_h
                x1 = c * self.patch_w
                x2 = x1 + self.patch_w

                canvas_img[y1:y2, x1:x2] = patch_img
                canvas_label[y1:y2, x1:x2] = patch_label

        # 裁到目标大小
        canvas_img = canvas_img[:self.out_h, :self.out_w]
        canvas_label = canvas_label[:self.out_h, :self.out_w]

        # 再保险一次，确保输出尺寸固定
        canvas_img, canvas_label = self._resize_pair(
            canvas_img, canvas_label, self.out_h, self.out_w
        )

        return canvas_img, canvas_label

    def __getitem__(self, idx):
        img, gt = self._compose_large_image()

        fake_name = f'syn_{idx}.png'

        results = dict(
            img_info=dict(filename=fake_name),
            ann_info=dict(seg_map=fake_name),

            # 这几个顶层字段必须补
            filename=fake_name,
            ori_filename=fake_name,

            img=img,
            gt_semantic_seg=gt,
        )

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
