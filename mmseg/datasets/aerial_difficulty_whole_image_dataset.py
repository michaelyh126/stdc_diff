import os
import random

import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data import Dataset

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose
from mmseg.utils import get_root_logger


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


@DATASETS.register_module()
class AerialDifficultyWholeImageDataset(Dataset):
    """Sample whole aerial images from difficulty folders.

    This dataset is for difficulty-ranked original images, not 500x500 patch
    composition. It samples one whole image/mask pair by difficulty ratio, then
    lets the normal segmentation pipeline load, crop, augment, and format it.

    Expected layout:

        data_root/
            easy/imgs/train/*.png
            easy/labels/train/*.png
            medium/imgs/train/*.png
            medium/labels/train/*.png
            hard/imgs/train/*.png
            hard/labels/train/*.png
    """

    CLASSES = ('background', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self,
                 data_root,
                 pipeline,
                 img_subdir='imgs/train',
                 ann_subdir='labels/train',
                 difficulty_names=('easy', 'medium', 'hard'),
                 sampling_ratios=(1, 2, 7),
                 img_suffix='',
                 seg_map_suffix='',
                 epoch_len=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 **kwargs):
        if len(difficulty_names) != len(sampling_ratios):
            raise ValueError('difficulty_names and sampling_ratios must have '
                             'the same length.')
        if any(r < 0 for r in sampling_ratios) or sum(sampling_ratios) <= 0:
            raise ValueError('sampling_ratios must be non-negative and have '
                             'a positive sum.')

        self.data_root = data_root
        self.pipeline = Compose(pipeline)
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.difficulty_names = tuple(difficulty_names)
        self.sampling_ratios = tuple(float(r) for r in sampling_ratios)
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.custom_classes = False
        self.label_map = None

        self.difficulty_infos = self._load_difficulty_infos()
        self.img_infos = [
            info for name in self.difficulty_names
            for info in self.difficulty_infos[name]
        ]
        if len(self.img_infos) == 0:
            raise ValueError(f'No image/label pairs found in {data_root}')

        self.epoch_len = int(epoch_len) if epoch_len is not None else len(
            self.img_infos)
        if self.epoch_len <= 0:
            raise ValueError('epoch_len must be positive.')

        self.sample_plan = self._build_sample_plan(self.epoch_len)
        self._log_dataset_info()

    def __len__(self):
        if self.test_mode:
            return len(self.img_infos)
        return self.epoch_len

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
                    filename=os.path.join(img_dir, image_name),
                    ann=dict(seg_map=os.path.join(ann_dir, label_name)),
                    difficulty=difficulty))

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
                    f'images were found.')
            difficulty_infos[difficulty] = infos
        return difficulty_infos

    def _build_sample_plan(self, length):
        ratios = np.array(self.sampling_ratios, dtype=np.float64)
        raw = ratios / ratios.sum() * length
        counts = np.floor(raw).astype(np.int64)
        remain = int(length - counts.sum())

        if remain > 0:
            order = sorted(
                range(len(ratios)),
                key=lambda i: (raw[i] - counts[i], ratios[i]),
                reverse=True)
            for idx in order[:remain]:
                counts[idx] += 1

        plan = []
        for difficulty, count in zip(self.difficulty_names, counts):
            plan.extend([difficulty] * int(count))
        random.shuffle(plan)
        return plan

    def _log_dataset_info(self):
        counts = {
            name: len(self.difficulty_infos[name])
            for name in self.difficulty_names
        }
        plan_counts = {
            name: self.sample_plan.count(name)
            for name in self.difficulty_names
        }
        msg = (
            f'AerialDifficultyWholeImageDataset loaded {len(self.img_infos)} '
            f'whole images, counts={counts}, ratios={self.sampling_ratios}, '
            f'epoch_len={self.epoch_len}, epoch_plan={plan_counts}')
        print_log(msg, logger=get_root_logger())

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = None
        results['seg_prefix'] = None
        if self.custom_classes:
            results['label_map'] = self.label_map

    def _prepare_info(self, info):
        results = dict(img_info=info, ann_info=info['ann'])
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_train_img(self, idx):
        difficulty = self.sample_plan[idx % len(self.sample_plan)]
        info = random.choice(self.difficulty_infos[difficulty])
        return self._prepare_info(info)

    def prepare_test_img(self, idx):
        info = self.img_infos[idx]
        return self._prepare_info(info)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        return self.prepare_train_img(idx)
