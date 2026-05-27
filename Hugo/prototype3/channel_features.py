from __future__ import annotations
from typing import Literal
import numpy as np


def extract_channel_matrix_features(
    H: np.ndarray,
    *,
    input_order: Literal["km", "mk"] = "mk",
    percentiles: tuple[float, ...] = (1, 5, 25, 50, 75, 95, 99),
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Extrae únicamente las características seleccionadas de la matriz compleja H.

    Características:
        - media del módulo
        - varianza del módulo
        - RMS del módulo
        - energía total del módulo
        - máximo del módulo
        - percentiles seleccionables
        - rango dinámico lineal del módulo
        - PLOS del PDP
        - energía total del PDP
        - sigma 1 de SVD
        - sigma 2 de SVD
        - spectral flatness de SVD
        - energía Doppler
    """

    # ============================================================
    # 0. Validaciones y orientación
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
        raise ValueError("H no contiene ningún valor complejo finito.")

    H_valid = H_km[valid_mask]

    features: dict[str, float] = {}

    # ============================================================
    # 1. Características del módulo |H|
    # ============================================================

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

    for p in percentiles:
        features[f"mag_percentile_{p:g}"] = float(np.percentile(A_valid, p))

    features["mag_dynamic_range_linear"] = float(A_max - A_min)

    # ============================================================
    # 2. PDP: Perfil de Retardo de Potencia
    # ============================================================

    H_clean_for_pdp = np.nan_to_num(
        H_km,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.complex128)

    h_tau = np.fft.ifft(H_clean_for_pdp, axis=0)
    pdp = np.abs(h_tau) ** 2

    pdp_mean = np.mean(pdp, axis=1)

    los_peak_idx = int(np.argmax(pdp_mean))
    los_peak_power = pdp_mean[los_peak_idx]
    pdp_total_energy = np.sum(pdp_mean)

    features["pdp_los_peak_power"] = float(los_peak_power)
    features["pdp_total_energy"] = float(pdp_total_energy)

    # ============================================================
    # 3. SVD
    # ============================================================

    H_clean_for_svd = np.nan_to_num(
        H_km,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.complex128)

    try:
        _, S, _ = np.linalg.svd(H_clean_for_svd, full_matrices=False)

        features["svd_sigma_1"] = float(S[0]) if len(S) > 0 else 0.0
        features["svd_sigma_2"] = float(S[1]) if len(S) > 1 else 0.0

        if len(S) > 0:
            geom_mean_S = np.exp(np.mean(np.log(S + eps)))
            arith_mean_S = np.mean(S)

            features["svd_spectral_flatness"] = float(
                geom_mean_S / (arith_mean_S + eps)
            )
        else:
            features["svd_spectral_flatness"] = 0.0

    except np.linalg.LinAlgError:
        features["svd_sigma_1"] = 0.0
        features["svd_sigma_2"] = 0.0
        features["svd_spectral_flatness"] = 0.0

    # ============================================================
    # 4. Energía Doppler
    # ============================================================

    if M >= 2:
        H_clean_for_doppler = np.nan_to_num(
            H_km,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.complex128)

        # Eliminación de componente media en slow-time
        H_ac = H_clean_for_doppler - np.mean(
            H_clean_for_doppler,
            axis=1,
            keepdims=True,
        )

        doppler_spectrum = np.abs(np.fft.fft(H_ac, axis=1)) ** 2
        doppler_mean = np.mean(doppler_spectrum, axis=0)

        features["doppler_variance_energy"] = float(np.sum(doppler_mean))
    else:
        features["doppler_variance_energy"] = 0.0

    return features


def print_channel_matrix_features(
    features: dict[str, float],
    *,
    title: str = "Características seleccionadas de la matriz de canal H",
    decimals: int = 6,
) -> None:
    """
    Imprime de forma ordenada las características seleccionadas.
    """

    def fmt_value(value):
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"

        if isinstance(value, (int, np.integer)):
            return str(int(value))

        return str(value)

    groups = {
        "Módulo de H": [
            "mag_mean",
            "mag_var",
            "mag_rms",
            "mag_energy_total",
            "mag_max",
            "mag_dynamic_range_linear",
        ],
        "Percentiles del módulo": [
            key for key in features.keys()
            if key.startswith("mag_percentile_")
        ],
        "Perfil de Retardo de Potencia (PDP)": [
            "pdp_los_peak_power",
            "pdp_total_energy",
        ],
        "Características SVD": [
            "svd_sigma_1",
            "svd_sigma_2",
            "svd_spectral_flatness",
        ],
        "Características Doppler": [
            "doppler_variance_energy",
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
            print(f"{key:<35}: {fmt_value(features[key])}")
            printed_keys.add(key)

    remaining_keys = [key for key in features.keys() if key not in printed_keys]

    if remaining_keys:
        print()
        print("[Otras características]")
        print("-" * 80)

        for key in remaining_keys:
            print(f"{key:<35}: {fmt_value(features[key])}")

    print()
    print("=" * 80)