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
    return_phase_arrays: bool = False,
    eps: float = 1e-12,
) -> dict[str, float | np.ndarray]:
    """
    Extrae características de una matriz compleja de canal H.

    La matriz H puede estar organizada como:

        input_order="km":
            H.shape = (N, M)
            H[k, m]

        input_order="mk":
            H.shape = (M, N)
            H[m, k]

    Internamente se convierte siempre a:

        H_km[k, m]

    Características extraídas
    -------------------------
    1. Módulo |H|:
        - media
        - varianza
        - RMS
        - energía total
        - máximo
        - mínimo
        - percentiles
        - rango dinámico lineal
        - rango dinámico en dB
        - entropía
        - kurtosis

    2. Fase arg(H):
        - media circular
        - varianza circular
        - coherencia de fase
        - fase desempaquetada en m
        - fase desempaquetada en k
        - saltos de fase en m
        - saltos de fase en k

    3. Perfil de Retardo de Potencia (PDP):
        - potencia del pico principal (LOS)
        - índice del pico principal
        - energía total del PDP
        - relación Pico-Promedio (PAPR del canal)

    4. Características del Subespacio (SVD):
        - valores singulares principales
        - ratio de energía en el primer modo
        - planitud espectral de los valores singulares

    5. Características Doppler:
        - energía de las variaciones temporales
        - frecuencia Doppler media ponderada

    Salida
    ------
    features : dict
        Diccionario con características escalares.
        Si return_phase_arrays=True, también incluye matrices auxiliares.
    """

    # ============================================================
    # 0. Validaciones y orientación de la matriz
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

    # H_km tiene shape (N, M)
    N, M = H_km.shape

    # Máscara de valores complejos válidos
    valid_mask = np.isfinite(H_km.real) & np.isfinite(H_km.imag)

    if not np.any(valid_mask):
        raise ValueError("H no contiene ningún valor complejo finito.")

    H_valid = H_km[valid_mask]

    features: dict[str, float | np.ndarray] = {}

    features["N_subcarriers"] = float(N)
    features["M_symbols"] = float(M)
    features["num_valid"] = float(np.sum(valid_mask))
    features["valid_ratio"] = float(np.mean(valid_mask))

    # ============================================================
    # 1. Características del módulo |H|
    # ============================================================

    A = np.abs(H_km)
    A_valid = np.abs(H_valid)

    A_valid = A_valid[np.isfinite(A_valid)]

    if A_valid.size == 0:
        raise ValueError("No hay valores finitos en |H|.")

    A_mean = np.mean(A_valid)
    A_var = np.var(A_valid)
    A_rms = np.sqrt(np.mean(A_valid**2))
    A_energy = np.sum(A_valid**2)
    A_max = np.max(A_valid)
    A_min = np.min(A_valid)

    features["mag_mean"] = float(A_mean)
    features["mag_var"] = float(A_var)
    features["mag_rms"] = float(A_rms)
    features["mag_energy_total"] = float(A_energy)
    features["mag_max"] = float(A_max)
    features["mag_min"] = float(A_min)

    # Percentiles
    for p in percentiles:
        features[f"mag_percentile_{p:g}"] = float(np.percentile(A_valid, p))

    # Rango dinámico lineal
    features["mag_dynamic_range_linear"] = float(A_max - A_min)

    # Rango dinámico en dB usando el menor valor positivo
    A_positive = A_valid[A_valid > eps]

    if A_positive.size > 0:
        A_min_positive = np.min(A_positive)
        dynamic_range_db = 20.0 * np.log10((A_max + eps) / (A_min_positive + eps))
    else:
        dynamic_range_db = 0.0

    features["mag_dynamic_range_db"] = float(dynamic_range_db)

    # Entropía del módulo mediante histograma normalizado
    hist, _ = np.histogram(A_valid, bins=entropy_bins, density=False)
    prob = hist.astype(np.float64)
    prob = prob / (np.sum(prob) + eps)
    prob = prob[prob > 0]

    mag_entropy = -np.sum(prob * np.log2(prob + eps))
    features["mag_entropy"] = float(mag_entropy)

    # Entropía normalizada entre 0 y 1
    if entropy_bins > 1:
        features["mag_entropy_normalized"] = float(
            mag_entropy / np.log2(entropy_bins)
        )
    else:
        features["mag_entropy_normalized"] = 0.0

    # Kurtosis del módulo
    A_std = np.std(A_valid)

    if A_std > eps:
        A_centered = A_valid - A_mean
        mag_kurtosis = np.mean((A_centered / A_std) ** 4)
        mag_excess_kurtosis = mag_kurtosis - 3.0
    else:
        mag_kurtosis = 0.0
        mag_excess_kurtosis = 0.0

    features["mag_kurtosis"] = float(mag_kurtosis)
    features["mag_excess_kurtosis"] = float(mag_excess_kurtosis)

    # ============================================================
    # 2. Características de fase arg(H)
    # ============================================================

    phase = np.angle(H_km)

    phase_valid = phase[valid_mask]
    phase_valid = phase_valid[np.isfinite(phase_valid)]

    if phase_valid.size == 0:
        raise ValueError("No hay valores finitos en la fase de H.")

    # ------------------------------------------------------------
    # 2.1 Media circular, varianza circular y coherencia
    # ------------------------------------------------------------

    z = np.exp(1j * phase_valid)

    mean_vector = np.mean(z)
    phase_coherence = np.abs(mean_vector)
    circular_mean = np.angle(mean_vector)
    circular_variance = 1.0 - phase_coherence

    features["phase_circular_mean"] = float(circular_mean)
    features["phase_circular_variance"] = float(circular_variance)
    features["phase_coherence"] = float(phase_coherence)

    # ------------------------------------------------------------
    # 2.2 Fase desempaquetada en m y en k
    # ------------------------------------------------------------
    # H_km[k, m]
    # axis=1 -> desempaquetado a lo largo de m
    # axis=0 -> desempaquetado a lo largo de k

    phase_unwrap_m = np.unwrap(phase, axis=1)
    phase_unwrap_k = np.unwrap(phase, axis=0)

    # Gradientes de fase desempaquetada
    dphase_dm = np.diff(phase_unwrap_m, axis=1)
    dphase_dk = np.diff(phase_unwrap_k, axis=0)

    dphase_dm_valid = dphase_dm[
        np.isfinite(dphase_dm)
    ]

    dphase_dk_valid = dphase_dk[
        np.isfinite(dphase_dk)
    ]

    if dphase_dm_valid.size > 0:
        features["phase_unwrap_m_mean"] = float(np.mean(phase_unwrap_m[np.isfinite(phase_unwrap_m)]))
        features["phase_unwrap_m_var"] = float(np.var(phase_unwrap_m[np.isfinite(phase_unwrap_m)]))
        features["phase_unwrap_m_slope_mean"] = float(np.mean(dphase_dm_valid))
        features["phase_unwrap_m_slope_var"] = float(np.var(dphase_dm_valid))
        features["phase_unwrap_m_slope_rms"] = float(
            np.sqrt(np.mean(dphase_dm_valid**2))
        )
    else:
        features["phase_unwrap_m_mean"] = 0.0
        features["phase_unwrap_m_var"] = 0.0
        features["phase_unwrap_m_slope_mean"] = 0.0
        features["phase_unwrap_m_slope_var"] = 0.0
        features["phase_unwrap_m_slope_rms"] = 0.0

    if dphase_dk_valid.size > 0:
        features["phase_unwrap_k_mean"] = float(np.mean(phase_unwrap_k[np.isfinite(phase_unwrap_k)]))
        features["phase_unwrap_k_var"] = float(np.var(phase_unwrap_k[np.isfinite(phase_unwrap_k)]))
        features["phase_unwrap_k_slope_mean"] = float(np.mean(dphase_dk_valid))
        features["phase_unwrap_k_slope_var"] = float(np.var(dphase_dk_valid))
        features["phase_unwrap_k_slope_rms"] = float(
            np.sqrt(np.mean(dphase_dk_valid**2))
        )
    else:
        features["phase_unwrap_k_mean"] = 0.0
        features["phase_unwrap_k_var"] = 0.0
        features["phase_unwrap_k_slope_mean"] = 0.0
        features["phase_unwrap_k_slope_var"] = 0.0
        features["phase_unwrap_k_slope_rms"] = 0.0

    # ============================================================
    # 3. Saltos de fase
    # ============================================================
    # Saltos en fase envuelta, antes de unwrap.
    # El salto se mide como diferencia angular principal en [-pi, pi].

    phase_diff_m_wrapped = np.angle(
        np.exp(1j * np.diff(phase, axis=1))
    )

    phase_diff_k_wrapped = np.angle(
        np.exp(1j * np.diff(phase, axis=0))
    )

    abs_jump_m = np.abs(phase_diff_m_wrapped)
    abs_jump_k = np.abs(phase_diff_k_wrapped)

    abs_jump_m_valid = abs_jump_m[np.isfinite(abs_jump_m)]
    abs_jump_k_valid = abs_jump_k[np.isfinite(abs_jump_k)]

    # Saltos en m
    if abs_jump_m_valid.size > 0:
        features["phase_jump_m_mean_abs"] = float(np.mean(abs_jump_m_valid))
        features["phase_jump_m_var_abs"] = float(np.var(abs_jump_m_valid))
        features["phase_jump_m_max_abs"] = float(np.max(abs_jump_m_valid))
        features["phase_jump_m_rms_abs"] = float(
            np.sqrt(np.mean(abs_jump_m_valid**2))
        )
        features["phase_jump_m_count"] = float(
            np.sum(abs_jump_m_valid >= phase_jump_threshold)
        )
        features["phase_jump_m_ratio"] = float(
            np.mean(abs_jump_m_valid >= phase_jump_threshold)
        )
    else:
        features["phase_jump_m_mean_abs"] = 0.0
        features["phase_jump_m_var_abs"] = 0.0
        features["phase_jump_m_max_abs"] = 0.0
        features["phase_jump_m_rms_abs"] = 0.0
        features["phase_jump_m_count"] = 0.0
        features["phase_jump_m_ratio"] = 0.0

    # Saltos en k
    if abs_jump_k_valid.size > 0:
        features["phase_jump_k_mean_abs"] = float(np.mean(abs_jump_k_valid))
        features["phase_jump_k_var_abs"] = float(np.var(abs_jump_k_valid))
        features["phase_jump_k_max_abs"] = float(np.max(abs_jump_k_valid))
        features["phase_jump_k_rms_abs"] = float(
            np.sqrt(np.mean(abs_jump_k_valid**2))
        )
        features["phase_jump_k_count"] = float(
            np.sum(abs_jump_k_valid >= phase_jump_threshold)
        )
        features["phase_jump_k_ratio"] = float(
            np.mean(abs_jump_k_valid >= phase_jump_threshold)
        )
    else:
        features["phase_jump_k_mean_abs"] = 0.0
        features["phase_jump_k_var_abs"] = 0.0
        features["phase_jump_k_max_abs"] = 0.0
        features["phase_jump_k_rms_abs"] = 0.0
        features["phase_jump_k_count"] = 0.0
        features["phase_jump_k_ratio"] = 0.0

    # ============================================================
    # 4. Devolver matrices auxiliares si se desea
    # ============================================================

    if return_phase_arrays:
        features["phase"] = phase
        features["phase_unwrap_m"] = phase_unwrap_m
        features["phase_unwrap_k"] = phase_unwrap_k
        features["phase_diff_m_wrapped"] = phase_diff_m_wrapped
        features["phase_diff_k_wrapped"] = phase_diff_k_wrapped

    # ============================================================
    # 5. Características del Perfil de Retardo de Potencia (PDP)
    # ============================================================
    # Aplicar IFFT a lo largo de las subportadoras (axis=0)
    # H_km tiene shape (N, M)
    h_tau = np.fft.ifft(H_km, axis=0)
    pdp = np.abs(h_tau)**2
    
    # Promediar el PDP a lo largo de todos los símbolos
    pdp_mean = np.mean(pdp, axis=1)
    
    # Encontrar el pico principal (suele ser el LOS / trayecto directo)
    los_peak_idx = np.argmax(pdp_mean)
    los_peak_power = pdp_mean[los_peak_idx]
    
    features["pdp_los_peak_power"] = float(los_peak_power)
    features["pdp_los_peak_idx"] = float(los_peak_idx)
    
    # Energía total del PDP y relación Pico-Promedio (PAPR del canal)
    pdp_total_energy = np.sum(pdp_mean)
    features["pdp_total_energy"] = float(pdp_total_energy)
    features["pdp_papr_linear"] = float(los_peak_power / (np.mean(pdp_mean) + eps))

    # ============================================================
    # 6. Características del Subespacio (SVD)
    # ============================================================
    # SVD sobre la matriz de canal (N x M)
    # Rellena NaNs con 0 temporalmente si es necesario para linalg
    H_clean = np.nan_to_num(H_km, nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        # full_matrices=False para que sea más rápido
        U, S, Vh = np.linalg.svd(H_clean, full_matrices=False)
        
        features["svd_sigma_1"] = float(S[0])
        features["svd_sigma_2"] = float(S[1]) if len(S) > 1 else 0.0
        
        # Ratio de concentración de energía en el primer modo espacial
        features["svd_energy_ratio_mode1"] = float(S[0]**2 / (np.sum(S**2) + eps))
        
        # Planitud espectral de los valores singulares (indica dispersión del canal)
        geom_mean_S = np.exp(np.mean(np.log(S + eps)))
        arith_mean_S = np.mean(S)
        features["svd_spectral_flatness"] = float(geom_mean_S / (arith_mean_S + eps))
        
    except np.linalg.LinAlgError:
        # En caso de que la SVD no converja por datos corruptos
        features["svd_sigma_1"] = 0.0
        features["svd_sigma_2"] = 0.0
        features["svd_energy_ratio_mode1"] = 0.0
        features["svd_spectral_flatness"] = 0.0

    # ============================================================
    # 7. Características Doppler (Micro-movimientos / Convección)
    # ============================================================
    # FFT a lo largo de los símbolos (axis=1) para ver la variación temporal
    # Restamos la media para eliminar la componente DC (0 Hz estático)
    H_ac = H_km - np.mean(H_km, axis=1, keepdims=True)
    
    doppler_spectrum = np.abs(np.fft.fft(H_ac, axis=1))**2
    
    # Promedio del espectro Doppler sobre todas las subportadoras
    doppler_mean = np.mean(doppler_spectrum, axis=0)
    
    # Energía de las variaciones temporales (dispersión Doppler)
    doppler_total_energy = np.sum(doppler_mean)
    features["doppler_variance_energy"] = float(doppler_total_energy)
    
    # Frecuencia Doppler media ponderada (Centro de masa del espectro)
    freqs = np.fft.fftfreq(M)
    doppler_centroid = np.sum(np.abs(freqs) * doppler_mean) / (doppler_total_energy + eps)
    features["doppler_centroid"] = float(doppler_centroid)

    return features


def print_channel_matrix_features(
    features: dict,
    *,
    title: str = "Características de la matriz de canal H",
    decimals: int = 6,
    print_arrays: bool = False,
    max_array_elements: int = 8,
) -> None:
    """
    Imprime de forma ordenada las características extraídas de H.

    Parámetros
    ----------
    features:
        Diccionario devuelto por extract_channel_matrix_features(...).

    title:
        Título que se imprimirá al principio.

    decimals:
        Número de decimales para valores float.

    print_arrays:
        Si False, no imprime matrices completas como phase_unwrap_m.
        Si True, imprime un resumen de arrays.

    max_array_elements:
        Número máximo de valores que se muestran de cada array si print_arrays=True.
    """

    def fmt_value(value):
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"

        if isinstance(value, (int, np.integer)):
            return str(int(value))

        if isinstance(value, np.ndarray):
            if not print_arrays:
                return f"array shape={value.shape}, dtype={value.dtype}"

            flat = value.ravel()
            n_show = min(max_array_elements, flat.size)
            preview = flat[:n_show]

            preview_str = np.array2string(
                preview,
                precision=decimals,
                separator=", ",
                suppress_small=False,
            )

            return (
                f"array shape={value.shape}, dtype={value.dtype}, "
                f"primeros valores={preview_str}"
            )

        return str(value)

    groups = {
        "Información general": [
            "N_subcarriers",
            "M_symbols",
            "num_valid",
            "valid_ratio",
        ],
        "Módulo de H": [
            "mag_mean",
            "mag_var",
            "mag_rms",
            "mag_energy_total",
            "mag_max",
            "mag_min",
            "mag_dynamic_range_linear",
            "mag_dynamic_range_db",
            "mag_entropy",
            "mag_entropy_normalized",
            "mag_kurtosis",
            "mag_excess_kurtosis",
        ],
        "Percentiles del módulo": [
            key for key in features.keys()
            if key.startswith("mag_percentile_")
        ],
        "Fase de H": [
            "phase_circular_mean",
            "phase_circular_variance",
            "phase_coherence",
        ],
        "Fase desempaquetada en m": [
            "phase_unwrap_m_mean",
            "phase_unwrap_m_var",
            "phase_unwrap_m_slope_mean",
            "phase_unwrap_m_slope_var",
            "phase_unwrap_m_slope_rms",
        ],
        "Fase desempaquetada en k": [
            "phase_unwrap_k_mean",
            "phase_unwrap_k_var",
            "phase_unwrap_k_slope_mean",
            "phase_unwrap_k_slope_var",
            "phase_unwrap_k_slope_rms",
        ],
        "Saltos de fase en m": [
            "phase_jump_m_mean_abs",
            "phase_jump_m_var_abs",
            "phase_jump_m_max_abs",
            "phase_jump_m_rms_abs",
            "phase_jump_m_count",
            "phase_jump_m_ratio",
        ],
        "Saltos de fase en k": [
            "phase_jump_k_mean_abs",
            "phase_jump_k_var_abs",
            "phase_jump_k_max_abs",
            "phase_jump_k_rms_abs",
            "phase_jump_k_count",
            "phase_jump_k_ratio",
        ],
        "Arrays auxiliares": [
            "phase",
            "phase_unwrap_m",
            "phase_unwrap_k",
            "phase_diff_m_wrapped",
            "phase_diff_k_wrapped",
        ],
        "Perfil de Retardo de Potencia (PDP)": [
            "pdp_los_peak_power",
            "pdp_los_peak_idx",
            "pdp_total_energy",
            "pdp_papr_linear",
        ],
        "Características del Subespacio (SVD)": [
            "svd_sigma_1",
            "svd_sigma_2",
            "svd_energy_ratio_mode1",
            "svd_spectral_flatness",
        ],
        "Características Doppler": [
            "doppler_variance_energy",
            "doppler_centroid",
        ],
    }

    print("=" * 80)
    print(title)
    print("=" * 80)

    printed_keys = set()

    for group_name, keys in groups.items():
        existing_keys = [key for key in keys if key in features]

        if not existing_keys:
            continue

        print()
        print(f"[{group_name}]")
        print("-" * 80)

        for key in existing_keys:
            value = features[key]

            if isinstance(value, np.ndarray) and not print_arrays:
                print(f"{key:<35}: array shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key:<35}: {fmt_value(value)}")

            printed_keys.add(key)

    remaining_keys = [key for key in features.keys() if key not in printed_keys]

    if remaining_keys:
        print()
        print("[Otras características]")
        print("-" * 80)

        for key in remaining_keys:
            value = features[key]
            print(f"{key:<35}: {fmt_value(value)}")

    print()
    print("=" * 80)