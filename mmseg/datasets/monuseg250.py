import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MoNuSeg250Dataset(CustomDataset):
    """MoNuSeg 250x250 patch dataset."""

    CLASSES = ('background', 'nucleus')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(MoNuSeg250Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
