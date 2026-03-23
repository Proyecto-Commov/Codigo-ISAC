# clutter_filter.py
import numpy as np

from config import PipelineConfig
from signal_types import SlowTimeSeries


class ClutterFilter:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def apply(self, slow: SlowTimeSeries) -> SlowTimeSeries:
        x = np.array(slow.series, dtype=np.complex128, copy=True)

        if self.cfg.enable_dc_removal:
            x = x - np.mean(x, axis=1, keepdims=True)

        if self.cfg.enable_clutter_highpass:
            x = self._ema_highpass(x, alpha=self.cfg.clutter_alpha)

        metadata = dict(slow.metadata)
        metadata["clutter_filter"] = {
            "dc_removal": self.cfg.enable_dc_removal,
            "ema_highpass": self.cfg.enable_clutter_highpass,
            "alpha": self.cfg.clutter_alpha,
        }
        return SlowTimeSeries(series=x, labels=slow.labels, metadata=metadata)

    @staticmethod
    def _ema_highpass(x: np.ndarray, alpha: float = 0.98) -> np.ndarray:
        """
        Filtro paso alto simple: y = x - lowpass_ema(x)
        """
        y = np.empty_like(x)
        low = np.zeros(x.shape[0], dtype=np.complex128)

        for k in range(x.shape[1]):
            low = alpha * low + (1.0 - alpha) * x[:, k]
            y[:, k] = x[:, k] - low

        return y