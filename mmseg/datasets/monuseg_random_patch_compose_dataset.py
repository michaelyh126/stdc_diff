from mmseg.datasets.builder import DATASETS
from mmseg.datasets.crag_random_patch_compose_dataset import (
    CragRandomPatchComposeDataset,
)


@DATASETS.register_module()
class MoNuSegRandomPatchComposeDataset(CragRandomPatchComposeDataset):
    """Compose one large MoNuSeg sample from random patch images."""

    CLASSES = ('background', 'nucleus')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __getitem__(self, idx):
        img, gt = self._compose_large_image()
        return self._pack_results(img, gt, f'monuseg_patch_syn_{idx}.png')
