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

        # Frecuencia de muestreo en slow-time:
        # una observación Doppler por símbolo OFDM útil
        symbol_duration = (self.cfg.n_subcarriers + self.cfg.cp_len) / self.cfg.sample_rate
        fs_slow = 1.0 / symbol_duration

        window = get_window(stft_cfg.window, stft_cfg.nperseg)

        specs = []
        final_f = None
        final_t = None

        for row in x:
            f, t, Zxx = stft(
                row,
                fs=fs_slow,
                window=window,
                nperseg=stft_cfg.nperseg,
                noverlap=stft_cfg.noverlap,
                nfft=stft_cfg.nfft,
                detrend=stft_cfg.detrend,
                return_onesided=stft_cfg.return_onesided,
                boundary=None,
                padded=False,
                scaling=stft_cfg.scaling,
            )
            specs.append(np.abs(np.fft.fftshift(Zxx, axes=0)) ** 2)
            final_f = np.fft.fftshift(f)
            final_t = t

        spec = np.stack(specs, axis=0)  # [n_series, n_freqs, n_times]

        metadata = dict(slow.metadata)
        metadata["fs_slow"] = fs_slow
        metadata["spectrogram"] = {
            "window": stft_cfg.window,
            "nperseg": stft_cfg.nperseg,
            "noverlap": stft_cfg.noverlap,
            "nfft": stft_cfg.nfft,
        }

        return SpectrogramResult(
            freqs=final_f,
            times=final_t,
            spec=spec,
            metadata=metadata,
        )