# spectrogram_generator.py
import numpy as np
from scipy.signal import stft, get_window

from config import PipelineConfig
from signal_types import SlowTimeSeries, SpectrogramResult


class SpectrogramGenerator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def generate(self, slow: SlowTimeSeries) -> SpectrogramResult:
        x = slow.series
        stft_cfg = self.cfg.stft

        symbol_duration = (self.cfg.n_subcarriers + self.cfg.cp_len) / self.cfg.sample_rate
        fs_slow = 1.0 / symbol_duration

        specs = []
        final_f = None
        final_t = None

        for row in x:
            row = np.asarray(row)
            L = len(row)

            if L < 2:
                raise ValueError("La serie slow-time es demasiado corta para calcular STFT.")

            # Ajuste automático
            nperseg = min(stft_cfg.nperseg, L)
            noverlap = min(stft_cfg.noverlap, nperseg - 1)
            nfft = stft_cfg.nfft if stft_cfg.nfft is not None else nperseg
            nfft = max(nfft, nperseg)

            window = get_window(stft_cfg.window, nperseg)

            f, t, Zxx = stft(
                row,
                fs=fs_slow,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend=stft_cfg.detrend,
                return_onesided=stft_cfg.return_onesided,
                boundary=None,
                padded=False,
                scaling=stft_cfg.scaling,
            )

            specs.append(np.abs(np.fft.fftshift(Zxx, axes=0)) ** 2)
            final_f = np.fft.fftshift(f)
            final_t = t

        spec = np.stack(specs, axis=0)

        metadata = dict(slow.metadata)
        metadata["fs_slow"] = fs_slow
        metadata["spectrogram"] = {
            "window": stft_cfg.window,
            "nperseg_requested": stft_cfg.nperseg,
            "noverlap_requested": stft_cfg.noverlap,
            "nfft_requested": stft_cfg.nfft,
        }

        return SpectrogramResult(
            freqs=final_f,
            times=final_t,
            spec=spec,
            metadata=metadata,
        )