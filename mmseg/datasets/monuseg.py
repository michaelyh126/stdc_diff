import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MoNuSegDataset(CustomDataset):
    """MoNuSeg nuclei segmentation dataset.

    The expected layout is:
        imgs/train/*.tif
        imgs/test/*.tif
        labels/train/*.png
        labels/test/*.png

    Labels are binary masks where 0 is background and 1 is nucleus.
    """

    CLASSES = ('background', 'nucleus')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(MoNuSegDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
