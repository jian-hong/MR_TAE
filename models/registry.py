"""Model registry for all ablation and comparison variants."""

from typing import Dict, Type

from .base import BaseDenoiser
from .variants import (
    MWCNNBiGRU,
    MWCNNSwin,
    MRTAEFull,
    MRTAENoAttn,
    MRTAENoBiGRU,
    MRTAENoMTL,
    MRTAENoSwin,
    MRTAENoWavelet,
    UNetAttn,
    UNetBiGRU,
    UNetBiGRUSwin,
)


MODEL_REGISTRY: Dict[str, Type[BaseDenoiser]] = {
    MRTAEFull.MODEL_ID: MRTAEFull,
    MRTAENoBiGRU.MODEL_ID: MRTAENoBiGRU,
    MRTAENoSwin.MODEL_ID: MRTAENoSwin,
    MRTAENoAttn.MODEL_ID: MRTAENoAttn,
    MRTAENoMTL.MODEL_ID: MRTAENoMTL,
    MRTAENoWavelet.MODEL_ID: MRTAENoWavelet,
    MWCNNBiGRU.MODEL_ID: MWCNNBiGRU,
    MWCNNSwin.MODEL_ID: MWCNNSwin,
    UNetBiGRUSwin.MODEL_ID: UNetBiGRUSwin,
    UNetBiGRU.MODEL_ID: UNetBiGRU,
    UNetAttn.MODEL_ID: UNetAttn,
}
