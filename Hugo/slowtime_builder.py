# slowtime_builder.py
import numpy as np

from signal_types import SlowTimeSeries


class SlowTimeBuilder:
    """
    Reorganiza o reduce las series slow-time antes de la STFT.
    """

    def build(self, slow: SlowTimeSeries, mode: str = "per_pilot") -> SlowTimeSeries:
        x = slow.series

        if mode == "per_pilot":
            return slow

        if mode == "average_all":
            avg = np.mean(x, axis=0, keepdims=True)
            labels = np.array(["average_all"], dtype=object)
            metadata = dict(slow.metadata)
            metadata["slowtime_mode"] = mode
            return SlowTimeSeries(series=avg, labels=labels, metadata=metadata)

        raise ValueError(f"Modo no soportado: {mode}")