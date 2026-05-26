from __future__ import annotations
from typing import Literal
import numpy as np


def extract_channel_matrix_features(
    H: np.ndarray,
    *,
    input_order: Literal["km", "mk"] = "mk",
    percentiles: tuple[float, ...] = (1, 5, 25, 50, 75, 95, 99),
    phase_jump_threshold: float = np.pi,
    entropy_bins: int = 64,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Version David: incluye todas las variables del prototipo original mas:

        mag_skewness          -- asimetria de la distribucion de |H|
        pdp_mean_delay        -- retardo medio del PDP (en bins FFT)
        pdp_rms_delay_spread  -- dispersion de retardo RMS
        pdp_los_ratio         -- fraccion de energia en el pico LOS
        svd_sigma_ratio       -- cociente sigma_1 / sigma_2
        dH_dt_mean            -- variacion temporal media simbolo a simbolo
        dH_dt_max             -- variacion temporal maxima simbolo a simbolo
        doppler_spread        -- ancho de banda RMS del espectro Doppler
    """

    # ============================================================
    # 0. Validaciones y orientacion
    # ============================================================

    H = np.asarray(H)

    if H.ndim != 2:
        raise ValueError(f"H debe ser una matriz 2D, recibido shape={H.shape}")

    if not np.iscomplexobj(H):
        raise TypeError("H debe ser una matriz compleja.")

    if input_order == "km":
        H_km = H
    elif input_order == "mk":
        H_km = H.T
    else:
        raise ValueError("input_order debe ser 'km' o 'mk'.")

    N, M = H_km.shape

    valid_mask = np.isfinite(H_km.real) & np.isfinite(H_km.imag)

    if not np.any(valid_mask):
        raise ValueError("H no contiene ningun valor complejo finito.")

    H_valid = H_km[valid_mask]
    H_clean = np.nan_to_num(
        H_km, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.complex128)

    features: dict[str, float] = {}

    features["N_subcarriers"] = float(N)
    features["M_symbols"]     = float(M)
    features["num_valid"]     = float(np.sum(valid_mask))
    features["valid_ratio"]   = float(np.mean(valid_mask))

    # ============================================================
    # 1. Modulo |H|
    # ============================================================

    A_valid = np.abs(H_valid)
    A_valid = A_valid[np.isfinite(A_valid)]

    if A_valid.size == 0:
        raise ValueError("No hay valores finitos en |H|.")

    A_mean   = float(np.mean(A_valid))
    A_var    = float(np.var(A_valid))
    A_std    = float(np.std(A_valid))
    A_rms    = float(np.sqrt(np.mean(A_valid ** 2)))
    A_energy = float(np.sum(A_valid ** 2))
    A_max    = float(np.max(A_valid))
    A_min    = float(np.min(A_valid))

    features["mag_mean"]          = A_mean
    features["mag_var"]           = A_var
    features["mag_rms"]           = A_rms
    features["mag_energy_total"]  = A_energy
    features["mag_max"]           = A_max
    features["mag_min"]           = A_min

    for p in percentiles:
        features[f"mag_percentile_{p:g}"] = float(np.percentile(A_valid, p))

    features["mag_dynamic_range_linear"] = float(A_max - A_min)

    A_positive = A_valid[A_valid > eps]
    if A_positive.size > 0:
        features["mag_dynamic_range_db"] = float(
            20.0 * np.log10((A_max + eps) / (np.min(A_positive) + eps))
        )
    else:
        features["mag_dynamic_range_db"] = 0.0

    # Entropia del modulo
    hist, _ = np.histogram(A_valid, bins=entropy_bins, density=False)
    prob = hist.astype(np.float64)
    prob = prob / (prob.sum() + eps)
    prob_nz = prob[prob > 0]
    mag_entropy = float(-np.sum(prob_nz * np.log2(prob_nz + eps)))
    features["mag_entropy"]            = mag_entropy
    features["mag_entropy_normalized"] = float(
        mag_entropy / np.log2(entropy_bins) if entropy_bins > 1 else 0.0
    )

    # Kurtosis y exceso de kurtosis
    if A_std > eps:
        A_centered = A_valid - A_mean
        mag_kurtosis = float(np.mean((A_centered / A_std) ** 4))
        features["mag_kurtosis"]        = mag_kurtosis
        features["mag_excess_kurtosis"] = mag_kurtosis - 3.0
    else:
        features["mag_kurtosis"]        = 0.0
        features["mag_excess_kurtosis"] = 0.0

    # NUEVA: asimetria (skewness)
    if A_std > eps:
        features["mag_skewness"] = float(
            np.mean(((A_valid - A_mean) / A_std) ** 3)
        )
    else:
        features["mag_skewness"] = 0.0

    # ============================================================
    # 2. Fase arg(H)
    # ============================================================

    phase = np.angle(H_km)

    phase_valid = phase[valid_mask]
    phase_valid = phase_valid[np.isfinite(phase_valid)]

    if phase_valid.size == 0:
        raise ValueError("No hay valores finitos en la fase de H.")

    # 2.1 Estadisticas circulares
    z = np.exp(1j * phase_valid)
    mean_vector      = np.mean(z)
    phase_coherence  = float(np.abs(mean_vector))
    circular_mean    = float(np.angle(mean_vector))
    circular_var     = float(1.0 - phase_coherence)

    features["phase_circular_mean"]     = circular_mean
    features["phase_circular_variance"] = circular_var
    features["phase_coherence"]         = phase_coherence

    # 2.2 Fase desempaquetada en m (eje temporal, axis=1) y en k (frecuencia, axis=0)
    phase_unwrap_m = np.unwrap(phase, axis=1)
    phase_unwrap_k = np.unwrap(phase, axis=0)

    dphase_dm = np.diff(phase_unwrap_m, axis=1)
    dphase_dk = np.diff(phase_unwrap_k, axis=0)

    dphase_dm_v = dphase_dm[np.isfinite(dphase_dm)]
    dphase_dk_v = dphase_dk[np.isfinite(dphase_dk)]

    pum_v = phase_unwrap_m[np.isfinite(phase_unwrap_m)]
    puk_v = phase_unwrap_k[np.isfinite(phase_unwrap_k)]

    if dphase_dm_v.size > 0:
        features["phase_unwrap_m_mean"]       = float(np.mean(pum_v))
        features["phase_unwrap_m_var"]        = float(np.var(pum_v))
        features["phase_unwrap_m_slope_mean"] = float(np.mean(dphase_dm_v))
        features["phase_unwrap_m_slope_var"]  = float(np.var(dphase_dm_v))
        features["phase_unwrap_m_slope_rms"]  = float(np.sqrt(np.mean(dphase_dm_v ** 2)))
    else:
        for k in ("phase_unwrap_m_mean", "phase_unwrap_m_var",
                  "phase_unwrap_m_slope_mean", "phase_unwrap_m_slope_var",
                  "phase_unwrap_m_slope_rms"):
            features[k] = 0.0

    if dphase_dk_v.size > 0:
        features["phase_unwrap_k_mean"]       = float(np.mean(puk_v))
        features["phase_unwrap_k_var"]        = float(np.var(puk_v))
        features["phase_unwrap_k_slope_mean"] = float(np.mean(dphase_dk_v))
        features["phase_unwrap_k_slope_var"]  = float(np.var(dphase_dk_v))
        features["phase_unwrap_k_slope_rms"]  = float(np.sqrt(np.mean(dphase_dk_v ** 2)))
    else:
        for k in ("phase_unwrap_k_mean", "phase_unwrap_k_var",
                  "phase_unwrap_k_slope_mean", "phase_unwrap_k_slope_var",
                  "phase_unwrap_k_slope_rms"):
            features[k] = 0.0

    # 2.3 Saltos de fase (diferencias angulares en fase envuelta)
    abs_jump_m = np.abs(np.angle(np.exp(1j * np.diff(phase, axis=1))))
    abs_jump_k = np.abs(np.angle(np.exp(1j * np.diff(phase, axis=0))))

    ajm = abs_jump_m[np.isfinite(abs_jump_m)]
    ajk = abs_jump_k[np.isfinite(abs_jump_k)]

    if ajm.size > 0:
        features["phase_jump_m_mean_abs"] = float(np.mean(ajm))
        features["phase_jump_m_var_abs"]  = float(np.var(ajm))
        features["phase_jump_m_max_abs"]  = float(np.max(ajm))
        features["phase_jump_m_rms_abs"]  = float(np.sqrt(np.mean(ajm ** 2)))
        features["phase_jump_m_count"]    = float(np.sum(ajm >= phase_jump_threshold))
        features["phase_jump_m_ratio"]    = float(np.mean(ajm >= phase_jump_threshold))
    else:
        for k in ("phase_jump_m_mean_abs", "phase_jump_m_var_abs", "phase_jump_m_max_abs",
                  "phase_jump_m_rms_abs", "phase_jump_m_count", "phase_jump_m_ratio"):
            features[k] = 0.0

    if ajk.size > 0:
        features["phase_jump_k_mean_abs"] = float(np.mean(ajk))
        features["phase_jump_k_var_abs"]  = float(np.var(ajk))
        features["phase_jump_k_max_abs"]  = float(np.max(ajk))
        features["phase_jump_k_rms_abs"]  = float(np.sqrt(np.mean(ajk ** 2)))
        features["phase_jump_k_count"]    = float(np.sum(ajk >= phase_jump_threshold))
        features["phase_jump_k_ratio"]    = float(np.mean(ajk >= phase_jump_threshold))
    else:
        for k in ("phase_jump_k_mean_abs", "phase_jump_k_var_abs", "phase_jump_k_max_abs",
                  "phase_jump_k_rms_abs", "phase_jump_k_count", "phase_jump_k_ratio"):
            features[k] = 0.0

    # ============================================================
    # 3. PDP: Perfil de Retardo de Potencia
    # ============================================================

    h_tau    = np.fft.ifft(H_clean, axis=0)
    pdp      = np.abs(h_tau) ** 2
    pdp_mean = np.mean(pdp, axis=1)          # (N,)

    los_peak_idx     = int(np.argmax(pdp_mean))
    los_peak_power   = float(pdp_mean[los_peak_idx])
    pdp_total_energy = float(np.sum(pdp_mean))

    features["pdp_los_peak_power"] = los_peak_power
    features["pdp_los_peak_idx"]   = float(los_peak_idx)
    features["pdp_total_energy"]   = pdp_total_energy
    features["pdp_papr_linear"]    = float(
        los_peak_power / (float(np.mean(pdp_mean)) + eps)
    )

    # NUEVAS: retardo medio, dispersion de retardo RMS y ratio LOS
    tau        = np.arange(N, dtype=np.float64)
    pdp_norm   = pdp_total_energy + eps
    tau_mean   = float(np.sum(tau * pdp_mean) / pdp_norm)
    tau_rms_sq = float(np.sum((tau ** 2) * pdp_mean) / pdp_norm) - tau_mean ** 2
    features["pdp_mean_delay"]       = tau_mean
    features["pdp_rms_delay_spread"] = float(np.sqrt(max(tau_rms_sq, 0.0)))
    features["pdp_los_ratio"]        = float(los_peak_power / pdp_norm)

    # ============================================================
    # 4. SVD
    # ============================================================

    try:
        _, S, _ = np.linalg.svd(H_clean, full_matrices=False)

        features["svd_sigma_1"] = float(S[0]) if len(S) > 0 else 0.0
        features["svd_sigma_2"] = float(S[1]) if len(S) > 1 else 0.0

        # NUEVA: cociente sigma_1 / sigma_2
        if len(S) > 1 and S[1] > eps:
            features["svd_sigma_ratio"] = float(S[0] / S[1])
        elif len(S) > 0:
            features["svd_sigma_ratio"] = float(S[0] / eps)
        else:
            features["svd_sigma_ratio"] = 0.0

        features["svd_energy_ratio_mode1"] = float(
            S[0] ** 2 / (np.sum(S ** 2) + eps)
        ) if len(S) > 0 else 0.0

        if len(S) > 0:
            geom_mean_S  = np.exp(np.mean(np.log(S + eps)))
            arith_mean_S = float(np.mean(S))
            features["svd_spectral_flatness"] = float(
                geom_mean_S / (arith_mean_S + eps)
            )
        else:
            features["svd_spectral_flatness"] = 0.0

    except np.linalg.LinAlgError:
        for k in ("svd_sigma_1", "svd_sigma_2", "svd_sigma_ratio",
                  "svd_energy_ratio_mode1", "svd_spectral_flatness"):
            features[k] = 0.0

    # ============================================================
    # 5. NUEVA: variacion temporal dH/dt simbolo a simbolo
    # ============================================================

    if M >= 2:
        dH = H_clean[:, 1:] - H_clean[:, :-1]     # (N, M-1)
        dH_power = np.abs(dH) ** 2
        features["dH_dt_mean"] = float(np.mean(dH_power))
        features["dH_dt_max"]  = float(np.max(dH_power))
    else:
        features["dH_dt_mean"] = 0.0
        features["dH_dt_max"]  = 0.0

    # ============================================================
    # 6. Energia y dispersion Doppler
    # ============================================================

    if M >= 2:
        H_ac = H_clean - np.mean(H_clean, axis=1, keepdims=True)
        doppler_spectrum  = np.abs(np.fft.fft(H_ac, axis=1)) ** 2
        doppler_mean_spec = np.mean(doppler_spectrum, axis=0)   # (M,)

        doppler_total = float(np.sum(doppler_mean_spec))
        features["doppler_variance_energy"] = doppler_total

        # Centroide con frecuencias absolutas (igual que el prototipo)
        freqs_abs = np.abs(np.fft.fftfreq(M))
        features["doppler_centroid"] = float(
            np.sum(freqs_abs * doppler_mean_spec) / (doppler_total + eps)
        )

        # NUEVA: dispersion Doppler (ancho de banda RMS con frecuencias con signo)
        freqs_signed = np.fft.fftfreq(M)
        dc = float(np.sum(freqs_signed * doppler_mean_spec) / (doppler_total + eps))
        spread_sq = float(
            np.sum((freqs_signed ** 2) * doppler_mean_spec) / (doppler_total + eps)
        ) - dc ** 2
        features["doppler_spread"] = float(np.sqrt(max(spread_sq, 0.0)))
    else:
        features["doppler_variance_energy"] = 0.0
        features["doppler_centroid"]        = 0.0
        features["doppler_spread"]          = 0.0

    return features
