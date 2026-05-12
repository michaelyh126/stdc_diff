import os
import random
import math
import cv2
import mmcv
import numpy as np
import torch

from torch.utils.data import Dataset
from mmseg.datasets.builder import DATASETS


@DATASETS.register_module()
class CragMixedPairDataset(Dataset):
    """
    每次返回两张图：
    1. Crag 原始大图（resize到 out_size）
    2. 由 Crag510 随机patch拼接得到的大图（最终resize到 out_size）

    最终返回：
        img: [2, C, H, W]
        gt_semantic_seg: [2, H, W]
    """

    CLASSES = ('background', 'gland')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self,
                 crag_root,
                 crag510_root,
                 out_size=(1516, 1516),
                 patch_size=(510, 510),
                 img_suffix='.png',
                 seg_suffix='.png',
                 ignore_index=255,
                 photometric_distortion=False,
                 random_flip=False,
                 random_rotate=False):
        self.crag_root = crag_root
        self.crag510_root = crag510_root

        self.out_h, self.out_w = out_size
        self.patch_h, self.patch_w = patch_size
        self.ignore_index = ignore_index

        self.img_suffix = img_suffix
        self.seg_suffix = seg_suffix

        self.photometric_distortion = photometric_distortion
        self.random_flip = random_flip
        self.random_rotate = random_rotate

        self.crag_img_dir = os.path.join(crag_root, 'train/Images')
        self.crag_ann_dir = os.path.join(crag_root, 'train/Annotation')

        self.crag510_img_dir = os.path.join(crag510_root, 'train/Images')
        self.crag510_ann_dir = os.path.join(crag510_root, 'train/Annotation')

        self.crag_infos = self._load_file_pairs(self.crag_img_dir, self.crag_ann_dir)
        self.crag510_infos = self._load_file_pairs(self.crag510_img_dir, self.crag510_ann_dir)

        if len(self.crag_infos) == 0:
            raise ValueError(f'No files found in {self.crag_img_dir}')
        if len(self.crag510_infos) == 0:
            raise ValueError(f'No files found in {self.crag510_img_dir}')

    def _load_file_pairs(self, img_dir, ann_dir):
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(self.img_suffix.lower())
        ])

        infos = []
        for img_name in img_files:
            stem = os.path.splitext(img_name)[0]
            ann_name = stem + self.seg_suffix
            img_path = os.path.join(img_dir, img_name)
            ann_path = os.path.join(ann_dir, ann_name)
            if os.path.exists(ann_path):
                infos.append(dict(img_path=img_path, ann_path=ann_path, filename=img_name))
        return infos

    def __len__(self):
        return len(self.crag_infos)

    def _read_img_label(self, img_path, ann_path):
        img = mmcv.imread(img_path, flag='color')  # HWC, BGR
        label = mmcv.imread(ann_path, flag='unchanged')

        if label.ndim == 3:
            label = label[:, :, 0]

        return img, label

    def _remove_padding_by_ignore_index(self, img, label):
        """
        label==255 代表padding，要裁掉外围padding区域
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

    def _resize_img_label(self, img, label, out_h, out_w):
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return img, label

    def _random_augment(self, img, label):
        if self.random_flip:
            if random.random() < 0.5:
                img = np.flip(img, axis=1).copy()
                label = np.flip(label, axis=1).copy()

        if self.random_rotate:
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k, axes=(0, 1)).copy()
                label = np.rot90(label, k, axes=(0, 1)).copy()

        if self.photometric_distortion:
            # 简化版，避免依赖mmseg pipeline组件
            if random.random() < 0.5:
                alpha = random.uniform(0.9, 1.1)
                beta = random.uniform(-10, 10)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return img, label

    def _sample_valid_patch_from_crag510(self):
        """
        从Crag510中随机取一张patch，先去掉label中的255 padding，再resize回510x510
        """
        for _ in range(100):
            info = random.choice(self.crag510_infos)
            img, label = self._read_img_label(info['img_path'], info['ann_path'])

            img, label = self._remove_padding_by_ignore_index(img, label)
            if img is None:
                continue

            img, label = self._resize_img_label(img, label, self.patch_h, self.patch_w)
            img, label = self._random_augment(img, label)
            return img, label

        raise RuntimeError('Failed to sample a valid patch from Crag510 after many tries.')

    def _build_synthetic_large_image(self):
        """
        用Crag510里的多个patch拼成一张大图，然后resize到out_size
        """
        grid_rows = math.ceil(self.out_h / self.patch_h)
        grid_cols = math.ceil(self.out_w / self.patch_w)

        canvas_h = grid_rows * self.patch_h
        canvas_w = grid_cols * self.patch_w

        canvas_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas_label = np.full((canvas_h, canvas_w), self.ignore_index, dtype=np.uint8)

        for r in range(grid_rows):
            for c in range(grid_cols):
                patch_img, patch_label = self._sample_valid_patch_from_crag510()

                y1 = r * self.patch_h
                y2 = y1 + self.patch_h
                x1 = c * self.patch_w
                x2 = x1 + self.patch_w

                canvas_img[y1:y2, x1:x2] = patch_img
                canvas_label[y1:y2, x1:x2] = patch_label

        # 先裁到目标大小，再确保就是out_size
        canvas_img = canvas_img[:self.out_h, :self.out_w]
        canvas_label = canvas_label[:self.out_h, :self.out_w]

        # 理论上这里已经是目标大小，保留这步是为了稳妥
        canvas_img, canvas_label = self._resize_img_label(
            canvas_img, canvas_label, self.out_h, self.out_w
        )

        return canvas_img, canvas_label

    def _prepare_crag_large_image(self, idx):
        """
        读取Crag原始大图，并resize到out_size
        """
        info = self.crag_infos[idx]
        img, label = self._read_img_label(info['img_path'], info['ann_path'])

        # 如果原始label也有255 padding，可以裁掉；如果没有，这步不会出问题
        img2, label2 = self._remove_padding_by_ignore_index(img, label)
        if img2 is not None:
            img, label = img2, label2

        img, label = self._resize_img_label(img, label, self.out_h, self.out_w)
        img, label = self._random_augment(img, label)

        return img, label

    def _to_tensor_and_stack(self, img_big, label_big, img_syn, label_syn):
        # BGR -> RGB
        img_big = img_big[:, :, ::-1]
        img_syn = img_syn[:, :, ::-1]

        # HWC -> CHW
        img_big = torch.from_numpy(np.ascontiguousarray(img_big.transpose(2, 0, 1))).float()
        img_syn = torch.from_numpy(np.ascontiguousarray(img_syn.transpose(2, 0, 1))).float()

        label_big = torch.from_numpy(np.ascontiguousarray(label_big)).long()
        label_syn = torch.from_numpy(np.ascontiguousarray(label_syn)).long()

        # batch维拼接
        imgs = torch.stack([img_big, img_syn], dim=0)          # [2, C, H, W]
        labels = torch.stack([label_big, label_syn], dim=0)    # [2, H, W]

        return imgs, labels

    def __getitem__(self, idx):
        img_big, label_big = self._prepare_crag_large_image(idx)
        img_syn, label_syn = self._build_synthetic_large_image()

        imgs, labels = self._to_tensor_and_stack(
            img_big, label_big, img_syn, label_syn
        )

        results = dict(
            img=imgs,
            gt_semantic_seg=labels,
            img_metas=dict(
                filename=self.crag_infos[idx]['filename'],
                ori_shape=(self.out_h, self.out_w, 3),
                img_shape=(self.out_h, self.out_w, 3),
                pad_shape=(self.out_h, self.out_w, 3),
                scale_factor=1.0,
                flip=False
            )
        )
        return results
