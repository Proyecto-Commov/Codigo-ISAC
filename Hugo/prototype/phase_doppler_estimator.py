# phase_doppler_estimator.py
import numpy as np

from Hugo.prototype.signal_types import ChannelEstimate, SlowTimeSeries


class PhaseDopplerEstimator:
    """
    A partir de H[f, m, p], construye una señal slow-time compleja por piloto:
        s[m] = H[m+1] * conj(H[m])
    Esto elimina parte de la fase absoluta y resalta la evolución temporal.
    """

    def estimate(self, ch: ChannelEstimate) -> SlowTimeSeries:
        h = ch.h_pilots  # [F, M, P]

        if h.shape[1] < 2:
            raise ValueError("Se necesitan al menos 2 símbolos para diferencias de fase.")

        doppler_base = h[:, 1:, :] * np.conj(h[:, :-1, :])  # [F, M-1, P]

        # Reorganizamos a series independientes por (frame, pilot)
        # salida: [n_series, n_time]
        n_frames, n_time, n_pilots = doppler_base.shape
        series = np.transpose(doppler_base, (0, 2, 1)).reshape(n_frames * n_pilots, n_time)

        labels = []
        for f in range(n_frames):
            for p in ch.pilot_indices:
                labels.append((f, int(p)))
        labels = np.array(labels, dtype=object)

        metadata = dict(ch.metadata)
        metadata["doppler_representation"] = "consecutive_phase_product"
        return SlowTimeSeries(series=series, labels=labels, metadata=metadata)