from typing import Optional

from estimate.filters.oscillator_base import OscillatorParams
from estimate.filters.ossm_kf import oscillator_KFstd
from estimate.filters.parh_ossm import oscillator_PARH_OSSM

HEAD_REGISTRY = {
    "kfstd": oscillator_KFstd,
    "kf_std": oscillator_KFstd,
    "parh_ossm": oscillator_PARH_OSSM,
    "parhossm": oscillator_PARH_OSSM,
}


def build_head(head_key: str, params: Optional[OscillatorParams] = None):
    return HEAD_REGISTRY[head_key.lower()](params=params)
