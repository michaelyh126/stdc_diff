from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .lraspp_head import LRASPPHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .point_head import PointHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .segformer_head import SegformerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .setr_mla_head import SETRMLAHead
from .setr_up_head import SETRUPHead
from .uper_head import UPerHead
from .aspp_refine_head import RefineASPPHead
from .refine_decode_head import RefineBaseDecodeHead
from .isdhead import ISDHead
from .isd_diff_head import ISDDiffHead
from .diff_head import DiffHead
from .sdd_head import SddHead
from .isd_diff_fast_head import ISDDiffFastHead
from .stdc_diff_head import STDCDiffHead
from .stdc_in_head import STDCInHead
from .isd_pid_head import ISDPidHead
from .dual_diff_head import DualDiffHead
from .pid_un_head import PidUnHead
from .dual_distill_head import DualDistillHead
from .vit_guidance_head import VitGuidanceHead
from .single_diff_head import SingleDiffHead

__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
    'PointHead', 'APCHead', 'DMHead', 'LRASPPHead', 'SETRUPHead',
    'SETRMLAHead', 'SegformerHead', 'RefineASPPHead', 'RefineBaseDecodeHead',
    'ISDHead','DiffHead','ISDDiffHead','SddHead','ISDDiffFastHead','STDCDiffHead',
    'STDCInHead','ISDPidHead','DualDiffHead','PidUnHead','DualDistillHead','VitGuidanceHead','SingleDiffHead'
]
