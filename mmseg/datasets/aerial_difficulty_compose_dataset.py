import math
import os
import random

import cv2
import mmcv
import numpy as np
from mmcv.utils import print_log

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.utils import get_root_logger


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


@DATASETS.register_module()
class AerialDifficultyComposeDataset(CustomDataset):
    """Compose aerial training samples from difficulty-stratified patches.

    Expected data layout:

        data_root/
            easy/imgs/train/*.png
            easy/labels/train/*.png
            medium/imgs/train/*.png
            medium/labels/train/*.png
            hard/imgs/train/*.png
            hard/labels/train/*.png

    Each returned sample is a large canvas made from 500x500 tiles. Tile
    difficulty is sampled by ``sampling_ratios``. With 25 tiles and ratios
    ``(1, 2, 7)``, the exact per-canvas plan becomes 2 easy, 5 medium, and
    18 hard tiles.
    """

    CLASSES = ('background', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self,
                 data_root,
                 img_subdir='imgs/train',
                 ann_subdir='labels/train',
                 difficulty_names=('easy', 'medium', 'hard'),
                 sampling_ratios=(1, 2, 7),
                 img_suffix='',
                 seg_map_suffix='',
                 patch_size=(500, 500),
                 out_size=(2500, 2500),
                 epoch_len=None,
                 exact_tile_ratio=True,
                 label_value_map=None,
                 ignore_index=255,
                 pipeline=None,
                 **kwargs):
        if len(difficulty_names) != len(sampling_ratios):
            raise ValueError('difficulty_names and sampling_ratios must have '
                             'the same length.')
        if any(r < 0 for r in sampling_ratios) or sum(sampling_ratios) <= 0:
            raise ValueError('sampling_ratios must be non-negative and have '
                             'a positive sum.')

        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.difficulty_names = tuple(difficulty_names)
        self.sampling_ratios = tuple(float(r) for r in sampling_ratios)
        self.patch_h, self.patch_w = patch_size
        self.out_h, self.out_w = out_size
        self.epoch_len = epoch_len
        self.exact_tile_ratio = exact_tile_ratio
        self.label_value_map = self._normalize_label_value_map(
            label_value_map)
        self.has_random_flip = self._pipeline_has_random_flip(pipeline)

        main_difficulty = self.difficulty_names[-1]
        super().__init__(
            data_root=data_root,
            img_dir=os.path.join(main_difficulty, img_subdir),
            ann_dir=os.path.join(main_difficulty, ann_subdir),
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            pipeline=pipeline,
            **kwargs)

        self.grid_rows = math.ceil(self.out_h / self.patch_h)
        self.grid_cols = math.ceil(self.out_w / self.patch_w)
        self.tiles_per_sample = self.grid_rows * self.grid_cols
        self.difficulty_infos = self._load_difficulty_infos()
        self.img_infos = [
            info for name in self.difficulty_names
            for info in self.difficulty_infos[name]
        ]
        if self.epoch_len is None:
            self.epoch_len = len(self.img_infos)

        self._log_dataset_info()

    @staticmethod
    def _pipeline_has_random_flip(pipeline):
        if pipeline is None:
            return False
        return any(t.get('type') == 'RandomFlip' for t in pipeline
                   if isinstance(t, dict))

    @staticmethod
    def _normalize_label_value_map(label_value_map):
        if label_value_map is None:
            return None

        if isinstance(label_value_map, dict):
            items = label_value_map.items()
        else:
            items = label_value_map

        return [(int(src), int(dst)) for src, dst in items]

    def __len__(self):
        return int(self.epoch_len)

    def _is_valid_image_name(self, name):
        if self.img_suffix:
            return name.endswith(self.img_suffix)
        return name.lower().endswith(IMG_EXTS)

    def _label_name_from_image_name(self, image_name):
        if self.img_suffix:
            return image_name.replace(self.img_suffix, self.seg_map_suffix)
        return image_name

    def _load_one_difficulty(self, difficulty):
        img_dir = os.path.join(self.data_root, difficulty, self.img_subdir)
        ann_dir = os.path.join(self.data_root, difficulty, self.ann_subdir)

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f'Image dir does not exist: {img_dir}')
        if not os.path.isdir(ann_dir):
            raise FileNotFoundError(f'Label dir does not exist: {ann_dir}')

        label_names = set(mmcv.scandir(ann_dir, recursive=False))
        infos = []
        missing = []

        for image_name in sorted(mmcv.scandir(img_dir, recursive=False)):
            if not self._is_valid_image_name(image_name):
                continue

            label_name = self._label_name_from_image_name(image_name)
            if label_name not in label_names:
                missing.append(image_name)
                continue

            infos.append(
                dict(
                    filename=image_name,
                    ann=dict(seg_map=label_name),
                    difficulty=difficulty,
                    img_path=os.path.join(img_dir, image_name),
                    ann_path=os.path.join(ann_dir, label_name)))

        if missing:
            preview = ', '.join(missing[:10])
            raise FileNotFoundError(
                f'{len(missing)} images in {difficulty} have no matching '
                f'labels. First missing: {preview}')

        return infos

    def _load_difficulty_infos(self):
        difficulty_infos = {}
        for difficulty, ratio in zip(self.difficulty_names,
                                     self.sampling_ratios):
            infos = self._load_one_difficulty(difficulty)
            if ratio > 0 and len(infos) == 0:
                raise ValueError(
                    f'Difficulty "{difficulty}" has ratio {ratio}, but no '
                    f'patches were found.')
            difficulty_infos[difficulty] = infos
        return difficulty_infos

    def _log_dataset_info(self):
        counts = {
            name: len(self.difficulty_infos[name])
            for name in self.difficulty_names
        }
        tile_counts = self._tile_counts()
        msg = (
            f'AerialDifficultyComposeDataset loaded {len(self.img_infos)} '
            f'patches, counts={counts}, ratios={self.sampling_ratios}, '
            f'tiles_per_sample={self.tiles_per_sample}, '
            f'tile_counts={tile_counts}, epoch_len={self.epoch_len}')
        print_log(msg, logger=get_root_logger())

    def _tile_counts(self):
        ratios = np.array(self.sampling_ratios, dtype=np.float64)
        raw = ratios / ratios.sum() * self.tiles_per_sample
        counts = np.floor(raw).astype(np.int64)
        remain = int(self.tiles_per_sample - counts.sum())

        if remain > 0:
            order = sorted(
                range(len(ratios)),
                key=lambda i: (raw[i] - counts[i], ratios[i]),
                reverse=True)
            for idx in order[:remain]:
                counts[idx] += 1

        return {
            name: int(count)
            for name, count in zip(self.difficulty_names, counts)
        }

    def _build_tile_plan(self):
        if not self.exact_tile_ratio:
            return random.choices(
                self.difficulty_names,
                weights=self.sampling_ratios,
                k=self.tiles_per_sample)

        tile_counts = self._tile_counts()
        plan = []
        for difficulty in self.difficulty_names:
            plan.extend([difficulty] * tile_counts[difficulty])
        random.shuffle(plan)
        return plan

    def _read_pair(self, info):
        img = mmcv.imread(info['img_path'], flag='color')
        label = mmcv.imread(info['ann_path'], flag='unchanged')

        if img is None:
            raise FileNotFoundError(f'Cannot read image: {info["img_path"]}')
        if label is None:
            raise FileNotFoundError(f'Cannot read label: {info["ann_path"]}')

        if label.ndim == 3:
            label = label[:, :, 0]
        if self.label_value_map is not None:
            label = self._map_label_values(label)

        if label.shape[:2] != img.shape[:2]:
            label = cv2.resize(
                label,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST)

        return img, label

    def _map_label_values(self, label):
        mapped = label.copy()
        for src, dst in self.label_value_map:
            mapped[label == src] = dst
        return mapped

    def _resize_pair(self, img, label, out_h, out_w):
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(
            label, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return img, label

    def _crop_or_resize_patch(self, img, label):
        h, w = img.shape[:2]
        if h < self.patch_h or w < self.patch_w:
            img, label = self._resize_pair(
                img, label, max(h, self.patch_h), max(w, self.patch_w))
            h, w = img.shape[:2]

        if h == self.patch_h and w == self.patch_w:
            return img.copy(), label.copy()

        y1 = random.randint(0, h - self.patch_h)
        x1 = random.randint(0, w - self.patch_w)
        y2 = y1 + self.patch_h
        x2 = x1 + self.patch_w
        return img[y1:y2, x1:x2].copy(), label[y1:y2, x1:x2].copy()

    def _sample_one_patch(self, difficulty):
        infos = self.difficulty_infos[difficulty]
        info = random.choice(infos)
        img, label = self._read_pair(info)
        return self._crop_or_resize_patch(img, label)

    def _compose_large_image(self):
        canvas_h = self.grid_rows * self.patch_h
        canvas_w = self.grid_cols * self.patch_w
        canvas_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas_label = np.full(
            (canvas_h, canvas_w), self.ignore_index, dtype=np.uint8)
        plan = self._build_tile_plan()

        for tile_idx, difficulty in enumerate(plan):
            row = tile_idx // self.grid_cols
            col = tile_idx % self.grid_cols
            y1 = row * self.patch_h
            y2 = y1 + self.patch_h
            x1 = col * self.patch_w
            x2 = x1 + self.patch_w

            patch_img, patch_label = self._sample_one_patch(difficulty)
            canvas_img[y1:y2, x1:x2] = patch_img
            canvas_label[y1:y2, x1:x2] = patch_label

        return (canvas_img[:self.out_h, :self.out_w],
                canvas_label[:self.out_h, :self.out_w])

    def __getitem__(self, idx):
        img, gt = self._compose_large_image()
        fake_name = f'aerial_difficulty_compose_{idx}.png'

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
