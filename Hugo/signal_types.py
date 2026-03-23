# signal_types.py
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class SignalBlock:
    source_type: str                     # "sionna", "usrp", "generic"
    data: np.ndarray                     # IQ o rejilla OFDM
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OFDMGrid:
    grid: np.ndarray                     # shape: [n_frames, n_symbols, n_subcarriers]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PilotObservations:
    y_pilots: np.ndarray                 # shape: [n_frames, n_symbols_eff, n_pilots]
    x_pilots: np.ndarray                 # misma forma o broadcast compatible
    pilot_indices: np.ndarray
    symbol_indices: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelEstimate:
    h_pilots: np.ndarray                 # shape: [n_frames, n_symbols_eff, n_pilots]
    pilot_indices: np.ndarray
    symbol_indices: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlowTimeSeries:
    series: np.ndarray                   # shape: [n_series, n_time]
    labels: Optional[np.ndarray] = None  # nombres/índices de cada serie
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectrogramResult:
    freqs: np.ndarray
    times: np.ndarray
    spec: np.ndarray                     # shape: [n_series, n_freqs, n_times]
    metadata: Dict[str, Any] = field(default_factory=dict)