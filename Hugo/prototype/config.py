# config.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class PilotConfig:
    msync: int = 2                  # Símbolo full-band de sincronización (OpenISAC)
    pilot_spacing: int = 4          # P = {0, 4, 8, ...}
    zc_root: int = 1                # raíz Zadoff-Chu
    use_full_sync_symbol: bool = True

    def pilot_indices(self, n_subcarriers: int) -> np.ndarray:
        return np.arange(0, n_subcarriers, self.pilot_spacing, dtype=int)


@dataclass
class STFTConfig:
    window: str = "hann"            # "hann" o "hamming"
    nperseg: int = 256
    noverlap: int = 192
    nfft: Optional[int] = 512
    detrend: bool = False
    return_onesided: bool = False
    scaling: str = "spectrum"


@dataclass
class PipelineConfig:
    n_subcarriers: int = 1024
    cp_len: int = 128
    n_symbols_per_frame: int = 100
    sample_rate: float = 50e6
    carrier_freq: float = 3.5e9

    pilot: PilotConfig = field(default_factory=PilotConfig)
    stft: STFTConfig = field(default_factory=STFTConfig)

    # Si ya tienes la rejilla OFDM y no IQ crudo:
    input_is_frequency_grid: bool = False

    # Asunción simple para IQ crudo:
    # se empieza en frontera de trama y no hay CFO/SFO severos.
    assume_frame_aligned: bool = True

    # Preprocesado básico
    enable_dc_removal: bool = True
    enable_clutter_highpass: bool = True
    clutter_alpha: float = 0.98  # filtro de media exponencial para clutter lento