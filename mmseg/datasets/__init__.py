from .ade import ADE20KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .drive import DRIVEDataset
from .hrf import HRFDataset
from .pascal_context import PascalContextDataset, PascalContextDataset59
from .stare import STAREDataset
from .voc import PascalVOCDataset
from .deepglobe import DeepGlobeDataset
from .inria_aerial import InriaAerialDataset
from .road import RoadDataset
from .camvid import CamVidDataSet

__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
    'PascalContextDataset59', 'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset',
    'STAREDataset', 'DeepGlobeDataset', 'InriaAerialDataset','RoadDataset','CamVidDataSet'
]
