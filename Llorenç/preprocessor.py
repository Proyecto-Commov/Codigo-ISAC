# preprocessor.py
"""
Módulo de preprocesado avanzado para micro-Doppler OFDM.

Etapas:
  1. ClutterRemover  – Eliminación de clutter estático (objetos inmóviles).
  2. MicroDopplerSeparator – Separación de firmas humanas vs. térmicas.
  3. Preprocessor    – Orquestador que combina ambas etapas.

Convenciones de forma de array
--------------------------------
  slow-time series : [n_series, n_time]   (complejo)
  espectrograma    : [n_series, n_freqs, n_times] (real ≥ 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, stft, get_window


# ---------------------------------------------------------------------------
# Tipos auxiliares
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    # ── Eliminación de clutter ──────────────────────────────────────────────
    clutter_method: str = "ema"        # "dc_sub" | "ema" | "svd" | "all"
    ema_alpha: float = 0.98            # coef. para EMA high-pass
    svd_rank: int = 1                  # rango de la componente de clutter

    # ── Separación de fuentes ───────────────────────────────────────────────
    separation_method: str = "bandpass"  # "bandpass" | "energy_ratio" | "combined"
    sample_rate: float = 50e6
    n_subcarriers: int = 1024
    cp_len: int = 128

    # Frecuencias Doppler de corte para humanos [Hz], relativas a fs_slow
    human_doppler_low_hz: float = 0.5    # ~0.5 Hz → respiración lenta
    human_doppler_high_hz: float = 20.0  # ~20 Hz  → movimiento rápido de extremidades

    # Umbral de ratio energía banda-humana / energía total
    human_energy_ratio_threshold: float = 0.15

    # STFT para análisis interno de separación
    nperseg: int = 64
    noverlap: int = 48
    nfft: int = 128
    window: str = "hann"


@dataclass
class PreprocessResult:
    """Contiene las series slow-time en cada etapa del preprocesado."""
    raw: np.ndarray                         # entrada original [n_series, n_time]
    after_clutter: np.ndarray               # tras eliminar clutter
    human_component: np.ndarray             # componente humana aislada
    thermal_component: np.ndarray           # componente térmica (residuo)
    human_mask: np.ndarray                  # bool [n_series] – ¿serie tiene firma humana?
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Eliminación de clutter estático
# ---------------------------------------------------------------------------

class ClutterRemover:
    """
    Tres estrategias complementarias para suprimir el clutter estático
    (paredes, muebles, estructuras) en el dominio slow-time.

    Métodos
    -------
    dc_sub  : Sustracción directa de la media temporal → cancela la componente
              Doppler-cero exacta.
    ema     : Filtro paso-alto por media exponencial móvil. Más robusto ante
              variaciones lentas del canal (deriva térmica, CFO residual).
    svd     : Descomposición SVD de la matriz [n_series × n_time].  Los
              primeros ``svd_rank`` vectores singulares capturan la respuesta
              estática (alta correlación espacial) y se restan.
    all     : Aplica los tres en secuencia.
    """

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

    def remove(self, series: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Parameters
        ----------
        series : np.ndarray, shape [n_series, n_time], dtype complex
        
        Returns
        -------
        filtered : np.ndarray, misma forma
        info     : dict con métricas de supresión
        """
        x = np.array(series, dtype=np.complex128, copy=True)
        method = self.cfg.clutter_method
        info: Dict[str, Any] = {"method": method}

        if method in ("dc_sub", "all"):
            x, sub_info = self._dc_subtraction(x)
            info["dc_sub"] = sub_info

        if method in ("ema", "all"):
            x, ema_info = self._ema_highpass(x, self.cfg.ema_alpha)
            info["ema"] = ema_info

        if method in ("svd", "all"):
            x, svd_info = self._svd_clutter(x, self.cfg.svd_rank)
            info["svd"] = svd_info

        if method not in ("dc_sub", "ema", "svd", "all"):
            raise ValueError(f"clutter_method desconocido: '{method}'")

        return x, info

    # ── Métodos internos ────────────────────────────────────────────────────

    @staticmethod
    def _dc_subtraction(x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Resta la media temporal de cada serie → suprime Doppler cero.
        
        Es la operación más sencilla: H_clean[m] = H[m] - mean_m(H[m]).
        Perfecta cuando el canal estático no varía en el tiempo.
        """
        mean_power_before = float(np.mean(np.abs(x) ** 2))
        x_out = x - np.mean(x, axis=1, keepdims=True)
        mean_power_after = float(np.mean(np.abs(x_out) ** 2))
        suppression_db = 10 * np.log10(
            (mean_power_before + 1e-30) / (mean_power_after + 1e-30)
        )
        return x_out, {
            "power_before": mean_power_before,
            "power_after": mean_power_after,
            "suppression_dB": suppression_db,
        }

    @staticmethod
    def _ema_highpass(x: np.ndarray, alpha: float) -> Tuple[np.ndarray, Dict]:
        """
        Filtro paso-alto vía media exponencial móvil (EMA).

        La EMA estima la componente de baja frecuencia (clutter lento):
            low[k] = alpha * low[k-1] + (1-alpha) * x[k]
            y[k]   = x[k] - low[k]

        Un alpha cercano a 1 deja pasar solo variaciones rápidas (Doppler
        dinámico) y elimina el clutter estático y las derivas lentas.
        La frecuencia de corte aproximada es:
            f_c ≈ (1 - alpha) * fs_slow / (2*pi)
        """
        y = np.empty_like(x)
        low = np.zeros(x.shape[0], dtype=np.complex128)
        for k in range(x.shape[1]):
            low = alpha * low + (1.0 - alpha) * x[:, k]
            y[:, k] = x[:, k] - low
        return y, {"alpha": alpha}

    @staticmethod
    def _svd_clutter(x: np.ndarray, rank: int) -> Tuple[np.ndarray, Dict]:
        """
        Supresión de clutter por SVD.

        La respuesta de canal estática es altamente correlada entre series
        (todos los pilotos ven el mismo reflector fijo). Por lo tanto, ocupa
        los primeros valores singulares de la matriz X = U Σ Vᴴ.
        
        Reconstrucción del clutter de rango ``rank`` y sustracción:
            X_clutter = U[:, :rank] * Σ[:rank] * Vᴴ[:rank, :]
            X_clean   = X - X_clutter
        """
        U, s, Vh = np.linalg.svd(x, full_matrices=False)
        clutter = (U[:, :rank] * s[:rank]) @ Vh[:rank, :]
        x_out = x - clutter
        energy_removed = float(np.sum(s[:rank] ** 2) / (np.sum(s ** 2) + 1e-30))
        return x_out, {
            "rank": rank,
            "singular_values": s[:min(rank + 3, len(s))].tolist(),
            "energy_fraction_removed": energy_removed,
        }


# ---------------------------------------------------------------------------
# 2. Separación de fuentes micro-Doppler
# ---------------------------------------------------------------------------

class MicroDopplerSeparator:
    """
    Distingue las componentes de micro-Doppler humanas de las térmicas.

    Bases físicas
    -------------
    Humanos (caminar, respiración, gestos)
        - Respiración:  0.2–0.5 Hz (Doppler ~0.5–2 Hz con fc=3.5 GHz)
        - Caminar/torso: 1–3 Hz
        - Extremidades:  5–20 Hz (oscilación de brazos y piernas)
        → Espectro: bandas discretas, energía concentrada, **periódico**.

    Fuentes térmicas (convección de aire caliente)
        - Variaciones de índice de refracción del aire: muy lentas (<0.5 Hz)
        - Sin periodicidad clara; espectro **difuso y continuo**
        - Potencia muy baja, uniformemente distribuida en frecuencia
        → Espectro: ruido coloreado de baja frecuencia.

    Métodos implementados
    ---------------------
    bandpass      : Filtro paso-banda en [f_low, f_high] para aislar la banda
                    humana en tiempo (serie filtrada). El residuo es térmico.
    energy_ratio  : Calcula la fracción de energía en la banda humana para
                    cada serie. Si supera el umbral → firma humana detectada.
    combined      : Aplica bandpass + energy_ratio para clasificar y separar.
    """

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.fs_slow = self._compute_fs_slow()

    def _compute_fs_slow(self) -> float:
        symbol_duration = (self.cfg.n_subcarriers + self.cfg.cp_len) / self.cfg.sample_rate
        return 1.0 / symbol_duration

    def separate(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Parameters
        ----------
        series : np.ndarray [n_series, n_time], complex

        Returns
        -------
        human_component   : np.ndarray [n_series, n_time]
        thermal_component : np.ndarray [n_series, n_time]
        human_mask        : np.ndarray [n_series], bool
        info              : dict con métricas por serie
        """
        method = self.cfg.separation_method

        if method == "bandpass":
            return self._bandpass_separate(series)
        elif method == "energy_ratio":
            return self._energy_ratio_separate(series)
        elif method == "combined":
            return self._combined_separate(series)
        else:
            raise ValueError(f"separation_method desconocido: '{method}'")

    # ── Método 1: filtro paso-banda ─────────────────────────────────────────

    def _bandpass_separate(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Diseña un filtro Butterworth paso-banda centrado en la banda humana
        y lo aplica a cada serie slow-time con filtfilt (cero retardo).

        La componente thermal = series - human_component.
        La clasificación (human_mask) se basa en si la energía de la componente
        humana supera el umbral de ratio definido en la config.
        """
        nyq = self.fs_slow / 2.0
        low_n = self.cfg.human_doppler_low_hz / nyq
        high_n = self.cfg.human_doppler_high_hz / nyq
        low_n = np.clip(low_n, 1e-4, 0.999)
        high_n = np.clip(high_n, 1e-4, 0.999)

        if low_n >= high_n:
            raise ValueError(
                f"Banda humana inválida: [{low_n:.4f}, {high_n:.4f}] normalizada. "
                "Ajusta human_doppler_low/high_hz."
            )

        # Orden 4 Butterworth
        b, a = butter(4, [low_n, high_n], btype="bandpass")

        human = np.zeros_like(series)
        for i, row in enumerate(series):
            # filtfilt requiere señal real; filtramos parte real e imaginaria por separado
            human[i] = (
                filtfilt(b, a, row.real).astype(np.float64)
                + 1j * filtfilt(b, a, row.imag).astype(np.float64)
            )

        thermal = series - human

        # Clasificación por energía relativa
        energy_human = np.sum(np.abs(human) ** 2, axis=1)
        energy_total = np.sum(np.abs(series) ** 2, axis=1) + 1e-30
        ratio = energy_human / energy_total
        human_mask = ratio >= self.cfg.human_energy_ratio_threshold

        info = {
            "method": "bandpass",
            "fs_slow_hz": self.fs_slow,
            "band_hz": [self.cfg.human_doppler_low_hz, self.cfg.human_doppler_high_hz],
            "energy_ratios": ratio.tolist(),
            "human_mask": human_mask.tolist(),
            "n_human_series": int(human_mask.sum()),
        }
        return human, thermal, human_mask, info

    # ── Método 2: clasificación por ratio de energía espectral ─────────────

    def _energy_ratio_separate(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Calcula la STFT de cada serie y mide la fracción de energía espectral
        en la banda humana [f_low, f_high].

        Ventaja: usa información tiempo-frecuencia, más discriminante que el
        filtro temporal simple, pues la energía puede estar concentrada en
        ráfagas cortas (pasos, gestos).
        """
        n_series, n_time = series.shape
        nperseg = min(self.cfg.nperseg, n_time)
        noverlap = min(self.cfg.noverlap, nperseg - 1)
        nfft = max(self.cfg.nfft, nperseg)
        window = get_window(self.cfg.window, nperseg)

        human = np.zeros_like(series)
        thermal = np.zeros_like(series)
        ratios = np.zeros(n_series)

        for i, row in enumerate(series):
            f, t, Zxx = stft(
                row, fs=self.fs_slow, window=window,
                nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                return_onesided=False, boundary=None, padded=False,
            )
            f_shifted = np.fft.fftshift(f)
            Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
            power = np.abs(Zxx_shifted) ** 2

            # Máscara espectral para banda humana
            band = (np.abs(f_shifted) >= self.cfg.human_doppler_low_hz) & \
                   (np.abs(f_shifted) <= self.cfg.human_doppler_high_hz)

            e_band = float(np.sum(power[band]))
            e_total = float(np.sum(power)) + 1e-30
            ratios[i] = e_band / e_total

            # Reconstrucción: máscaras en espectrograma → ISTFT simplificada
            # (usamos filtro paso-banda para coherencia con tiempo)
            nyq = self.fs_slow / 2.0
            lo = np.clip(self.cfg.human_doppler_low_hz / nyq, 1e-4, 0.999)
            hi = np.clip(self.cfg.human_doppler_high_hz / nyq, 1e-4, 0.999)
            b, a = butter(4, [lo, hi], btype="bandpass")
            human[i] = (
                filtfilt(b, a, row.real) + 1j * filtfilt(b, a, row.imag)
            )

        thermal = series - human
        human_mask = ratios >= self.cfg.human_energy_ratio_threshold

        info = {
            "method": "energy_ratio",
            "fs_slow_hz": self.fs_slow,
            "band_hz": [self.cfg.human_doppler_low_hz, self.cfg.human_doppler_high_hz],
            "energy_ratios": ratios.tolist(),
            "human_mask": human_mask.tolist(),
            "n_human_series": int(human_mask.sum()),
            "threshold": self.cfg.human_energy_ratio_threshold,
        }
        return human, thermal, human_mask, info

    # ── Método 3: combinado ─────────────────────────────────────────────────

    def _combined_separate(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Combina paso-banda + análisis espectral de periodicidad.

        Adicionalmente comprueba periodicidad: las firmas humanas tienen picos
        espectrales discretos (caminata ~2 Hz, respiración ~0.3 Hz).
        Se calcula la kurtosis espectral: alto valor → picos pronunciados
        (probable humano); bajo valor → espectro plano (probable térmico).
        """
        human_bp, thermal_bp, mask_bp, info_bp = self._bandpass_separate(series)
        _, _, mask_er, info_er = self._energy_ratio_separate(series)

        # Periodicidad: kurtosis espectral en banda humana
        kurtosis_scores = self._spectral_kurtosis(human_bp)
        kurtosis_threshold = 1.5  # heurístico; ajustar con datos reales
        mask_kurtosis = kurtosis_scores >= kurtosis_threshold

        # Consenso: al menos 2 de 3 indicadores activos
        votes = mask_bp.astype(int) + mask_er.astype(int) + mask_kurtosis.astype(int)
        human_mask = votes >= 2

        human = human_bp
        thermal = series - human

        info = {
            "method": "combined",
            "bandpass": info_bp,
            "energy_ratio": info_er,
            "kurtosis_scores": kurtosis_scores.tolist(),
            "kurtosis_threshold": kurtosis_threshold,
            "votes": votes.tolist(),
            "human_mask": human_mask.tolist(),
            "n_human_series": int(human_mask.sum()),
        }
        return human, thermal, human_mask, info

    def _spectral_kurtosis(self, series: np.ndarray) -> np.ndarray:
        """
        Calcula la kurtosis del espectro de potencia de cada serie.
        Una firma periódica (humano) produce pocos picos con alta energía
        → distribución espectral con kurtosis > 3 (leptocúrtica).
        Un espectro plano (térmico/ruido) tiene kurtosis ≈ 1.8.
        """
        scores = np.zeros(len(series))
        for i, row in enumerate(series):
            psd = np.abs(np.fft.fft(row)) ** 2
            psd /= psd.sum() + 1e-30
            mu = np.sum(psd * np.arange(len(psd))) / len(psd)
            sigma2 = np.sum(psd * (np.arange(len(psd)) - mu) ** 2)
            if sigma2 > 1e-30:
                scores[i] = np.sum(psd * (np.arange(len(psd)) - mu) ** 4) / sigma2 ** 2
        return scores


# ---------------------------------------------------------------------------
# 3. Orquestador
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Orquesta ClutterRemover + MicroDopplerSeparator.

    Orden de operaciones
    --------------------
    1. Eliminar clutter → series sin la respuesta estática dominante.
    2. Separar componentes:
       - human_component  → para análisis de actividad humana
       - thermal_component → para detección de fuego/calor
    """

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.clutter_remover = ClutterRemover(cfg)
        self.separator = MicroDopplerSeparator(cfg)

    def run(self, slow_series: np.ndarray) -> PreprocessResult:
        """
        Parameters
        ----------
        slow_series : np.ndarray [n_series, n_time], complex

        Returns
        -------
        PreprocessResult con todas las etapas intermedias.
        """
        # Etapa 1
        after_clutter, clutter_info = self.clutter_remover.remove(slow_series)

        # Etapa 2
        human, thermal, human_mask, sep_info = self.separator.separate(after_clutter)

        return PreprocessResult(
            raw=slow_series,
            after_clutter=after_clutter,
            human_component=human,
            thermal_component=thermal,
            human_mask=human_mask,
            metadata={
                "clutter": clutter_info,
                "separation": sep_info,
                "config": {
                    "clutter_method": self.cfg.clutter_method,
                    "separation_method": self.cfg.separation_method,
                    "ema_alpha": self.cfg.ema_alpha,
                    "svd_rank": self.cfg.svd_rank,
                    "human_band_hz": [
                        self.cfg.human_doppler_low_hz,
                        self.cfg.human_doppler_high_hz,
                    ],
                    "energy_ratio_threshold": self.cfg.human_energy_ratio_threshold,
                },
            },
        )
