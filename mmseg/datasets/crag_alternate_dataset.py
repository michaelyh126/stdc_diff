import os
import math
import random
import cv2
import mmcv
import numpy as np

from torch.utils.data import Dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose


@DATASETS.register_module()
class CragAlternateDataset(Dataset):
    """
    偶数 idx: 返回 Crag 原始大图
    奇数 idx: 返回 Crag510 patch 拼接大图

    返回格式保持 mmseg 标准单样本格式。
    """

    CLASSES = ('background', 'gland')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self,
                 crag_root,
                 crag510_root,
                 crag_img_dir='train/Images',
                 crag_ann_dir='train/Annotation',
                 crag510_img_dir='train/Images',
                 crag510_ann_dir='train/Annotation',
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 patch_size=(510, 510),
                 out_size=(1516, 1516),
                 ignore_index=255,
                 pipeline=None):
        self.crag_root = crag_root
        self.crag510_root = crag510_root

        self.crag_img_dir = os.path.join(crag_root, crag_img_dir)
        self.crag_ann_dir = os.path.join(crag_root, crag_ann_dir)

        self.crag510_img_dir = os.path.join(crag510_root, crag510_img_dir)
        self.crag510_ann_dir = os.path.join(crag510_root, crag510_ann_dir)

        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix

        self.patch_h, self.patch_w = patch_size
        self.out_h, self.out_w = out_size
        self.ignore_index = ignore_index

        self.pipeline = Compose(pipeline)

        self.crag_infos = self._load_file_pairs(self.crag_img_dir, self.crag_ann_dir)
        self.crag510_infos = self._load_file_pairs(self.crag510_img_dir, self.crag510_ann_dir)

        if len(self.crag_infos) == 0:
            raise ValueError(f'No files found in {self.crag_img_dir}')
        if len(self.crag510_infos) == 0:
            raise ValueError(f'No files found in {self.crag510_img_dir}')

        # 让一个 epoch 长度按较大者来，奇偶各占一半左右
        self.length = 2 * max(len(self.crag_infos), len(self.crag510_infos))

    def __len__(self):
        return self.length

    def _load_file_pairs(self, img_dir, ann_dir):
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(self.img_suffix.lower())
        ])

        infos = []
        for img_name in img_files:
            stem = os.path.splitext(img_name)[0]
            ann_name = stem + self.seg_map_suffix

            img_path = os.path.join(img_dir, img_name)
            ann_path = os.path.join(ann_dir, ann_name)

            if os.path.exists(ann_path):
                infos.append(
                    dict(
                        filename=img_name,
                        img_path=img_path,
                        ann_path=ann_path
                    )
                )
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

    def _resize_pair(self, img, label, out_h, out_w):
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return img, label

    def _remove_patch_padding(self, img, label):
        """
        只对 Crag510 patch 去 padding。
        label == 255 代表 padding。
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

    def _sample_one_patch_from_crag510(self):
        for _ in range(100):
            info = random.choice(self.crag510_infos)
            img, label = self._read_img_label(info)

            # img, label = self._remove_patch_padding(img, label)
            if img is None:
                continue

            img, label = self._resize_pair(img, label, self.patch_h, self.patch_w)
            return img, label

        raise RuntimeError('Failed to sample valid patch from Crag510 after 100 tries.')

    def _compose_large_image_from_crag510(self):
        grid_rows = math.ceil(self.out_h / self.patch_h)
        grid_cols = math.ceil(self.out_w / self.patch_w)

        canvas_h = grid_rows * self.patch_h
        canvas_w = grid_cols * self.patch_w

        canvas_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas_label = np.full((canvas_h, canvas_w), self.ignore_index, dtype=np.uint8)

        for r in range(grid_rows):
            for c in range(grid_cols):
                patch_img, patch_label = self._sample_one_patch_from_crag510()

                y1 = r * self.patch_h
                y2 = y1 + self.patch_h
                x1 = c * self.patch_w
                x2 = x1 + self.patch_w

                canvas_img[y1:y2, x1:x2] = patch_img
                canvas_label[y1:y2, x1:x2] = patch_label

        canvas_img = canvas_img[:self.out_h, :self.out_w]
        canvas_label = canvas_label[:self.out_h, :self.out_w]

        canvas_img, canvas_label = self._resize_pair(
            canvas_img, canvas_label, self.out_h, self.out_w
        )

        return canvas_img, canvas_label

    def _get_crag_sample(self, idx):
        real_idx = (idx // 2) % len(self.crag_infos)
        info = self.crag_infos[real_idx]

        img, label = self._read_img_label(info)
        img, label = self._resize_pair(img, label, self.out_h, self.out_w)

        fake_name = f'crag_{real_idx}.png'
        return self._pack_results(img, label, fake_name)

    def _get_crag510_syn_sample(self, idx):
        img, label = self._compose_large_image_from_crag510()
        fake_name = f'crag510_syn_{idx}.png'
        return self._pack_results(img, label, fake_name)

    def _pack_results(self, img, gt, filename):
        results = dict(
            img_info=dict(filename=filename),
            ann_info=dict(seg_map=filename),
            filename=filename,
            ori_filename=filename,
            img=img,
            gt_semantic_seg=gt,
        )

        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['flip'] = False
        results['flip_direction'] = None
        results['img_fields'] = ['img']
        results['seg_fields'] = ['gt_semantic_seg']

        return self.pipeline(results)

    def __getitem__(self, idx):
        if idx % 2 == 0:
            return self._get_crag_sample(idx)
        else:
            return self._get_crag510_syn_sample(idx)
