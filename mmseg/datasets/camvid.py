import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CamVidDataSet(CustomDataset):
    """DRIVE dataset.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """

    CLASSES = (
        'unknown',  # 0
        'Animal',  # 1
        'Archway',  # 2
        'Bicyclist',  # 3
        'Bridge',  # 4
        'Building',  # 5
        'Car',  # 6
        'CartLuggagePram',  # 7
        'Child',  # 8
        'Column_Pole',  # 9
        'Fence',  # 10
        'LaneMkgsDriv',  # 11
        'LaneMkgsNonDriv',  # 12
        'Misc_Text',  # 13
        'MotorcycleScooter',  # 14
        'OtherMoving',  # 15
        'ParkingBlock',  # 16
        'Pedestrian',  # 17
        'Road',  # 18
        'RoadShoulder',  # 19
        'Sidewalk',  # 20
        'SignSymbol',  # 21
        'Sky',  # 22
        'SUVPickupTruck',  # 23
        'TrafficCone',  # 24
        'TrafficLight',  # 25
        'Train',  # 26
        'Tree',  # 27
        'Truck_Bus',  # 28
        'Tunnel',  # 29
        'VegetationMisc',  # 30
        'Void',  # 31
        'Wall'  # 32
    )

    PALETTE = [
        [0, 0, 0],  # Void
        [64, 128, 64],  # Animal
        [192, 0, 128],  # Archway
        [0, 128, 192],  # Bicyclist
        [0, 128, 64],  # Bridge
        [128, 0, 0],  # Building
        [64, 0, 128],  # Car
        [64, 0, 192],  # CartLuggagePram
        [192, 128, 64],  # Child
        [192, 192, 128],  # Column_Pole
        [64, 64, 128],  # Fence
        [128, 0, 192],  # LaneMkgsDriv
        [192, 0, 64],  # LaneMkgsNonDriv
        [128, 128, 64],  # Misc_Text
        [192, 0, 192],  # MotorcycleScooter
        [128, 64, 64],  # OtherMoving
        [64, 192, 128],  # ParkingBlock
        [64, 64, 0],  # Pedestrian
        [128, 64, 128],  # Road
        [128, 128, 192],  # RoadShoulder
        [0, 0, 192],  # Sidewalk
        [192, 128, 128],  # SignSymbol
        [128, 128, 128],  # Sky
        [64, 128, 192],  # SUVPickupTruck
        [0, 0, 64],  # TrafficCone
        [0, 64, 64],  # TrafficLight
        [192, 64, 128],  # Train
        [128, 128, 0],  # Tree
        [192, 128, 192],  # Truck_Bus
        [64, 0, 64],  # Tunnel
        [192, 192, 0],  # VegetationMisc
        [0, 0, 0],  # Unknown
        [64, 192, 0]  # Wall
    ]

    def __init__(self, **kwargs):
        super(CamVidDataSet, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_L.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
