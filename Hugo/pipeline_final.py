from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from isac.channel_estimator import (
    load_modulator_yaml,
    read_usrp_iq_bin,
    demodulate_ofdm_iq,
    estimate_channel_grid,
    IQStorageFormat,
)

from pipeline_part_2 import (
    pca_subcarriers_single_matrix,
    extract_signal_features,
)


FeatureBranch = Literal["logmag", "dphi"]
ReductionMode = Literal["pca", "pilots"]


# =============================================================================
# Funciones auxiliares
# =============================================================================

def _sanitize_real_matrix(
    X: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Convierte una matriz real a float64 y sustituye NaN/inf por fill_value.
    """
    X = np.asarray(X, dtype=np.float64)

    return np.nan_to_num(
        X,
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    )


def _clean_complex_matrix(H: np.ndarray) -> np.ndarray:
    """
    Convierte H a matriz compleja y sustituye NaN/inf por 0+0j.
    Funciona aunque H llegue como real o como compleja.
    """
    H = np.asarray(H, dtype=np.complex128)

    mask = np.isfinite(H.real) & np.isfinite(H.imag)

    H_clean = np.zeros_like(H, dtype=np.complex128)
    H_clean[mask] = H[mask]

    return H_clean


def _compute_logmag_from_H(
    H: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Calcula log|H| a partir de H complejo.

    Entrada:
        H.shape = (M, N)

    Salida:
        logmag.shape = (M, N)
    """
    H_clean = _clean_complex_matrix(H)

    logmag = np.log(np.abs(H_clean) + eps)

    return _sanitize_real_matrix(logmag)


def _compute_dphi_from_H(
    H: np.ndarray,
    *,
    keep_length: bool = False,
) -> np.ndarray:
    """
    Calcula la diferencia de fase temporal:

        dphi[m,k] = angle(H[m+1,k] * conj(H[m,k]))

    Entrada:
        H.shape = (M, N)

    Salida:
        si keep_length=False:
            dphi.shape = (M-1, N)

        si keep_length=True:
            dphi.shape = (M, N)
    """
    H_clean = _clean_complex_matrix(H)

    if H_clean.shape[0] < 2:
        raise ValueError("H debe tener al menos 2 símbolos OFDM para calcular dphi.")

    dphi = np.angle(H_clean[1:, :] * np.conj(H_clean[:-1, :]))

    if keep_length:
        first_row = np.zeros((1, dphi.shape[1]), dtype=np.float64)
        dphi = np.vstack([first_row, dphi])

    return _sanitize_real_matrix(dphi)


def _get_slow_time_fs_from_cfg(
    cfg: dict[str, Any],
    *,
    symbol_stride: int = 1,
) -> float:
    """
    Frecuencia de muestreo slow-time de los símbolos OFDM.

        fs_slow = sample_rate / (fft_size + cp_length)
    """
    sample_rate = float(cfg["sample_rate"])
    fft_size = int(cfg["fft_size"])
    cp_length = int(cfg["cp_length"])

    if symbol_stride <= 0:
        raise ValueError("symbol_stride debe ser positivo.")

    samples_per_symbol = fft_size + cp_length

    return sample_rate / samples_per_symbol / symbol_stride


def _get_pilot_positions_from_cfg(
    cfg: dict[str, Any],
    *,
    fft_size: int,
) -> np.ndarray:
    """
    Lee pilot_positions del YAML y valida que estén dentro de [0, fft_size-1].
    """
    if "pilot_positions" not in cfg:
        raise ValueError("El YAML no contiene 'pilot_positions'.")

    pilot_positions = np.asarray(cfg["pilot_positions"], dtype=int).reshape(-1)

    if pilot_positions.size == 0:
        raise ValueError("'pilot_positions' está vacío.")

    if np.any(pilot_positions < 0) or np.any(pilot_positions >= fft_size):
        raise ValueError(
            f"Hay pilot_positions fuera de rango [0, {fft_size - 1}]: "
            f"{pilot_positions.tolist()}"
        )

    return pilot_positions


def _select_pilot_subcarriers_from_matrix(
    H_real: np.ndarray,
    *,
    pilot_positions: np.ndarray,
    n_subcarriers: int,
    auto_transpose: bool = True,
) -> np.ndarray:
    """
    Selecciona las columnas correspondientes a las subportadoras piloto.

    Entrada esperada:
        H_real.shape = (M, N)

    Salida:
        H_pilots.shape = (M, len(pilot_positions))
    """
    H_real = np.asarray(H_real, dtype=np.float64)

    if H_real.ndim != 2:
        raise ValueError(f"H_real debe ser 2D, recibido shape={H_real.shape}")

    if H_real.shape[1] == n_subcarriers:
        X = H_real
    elif auto_transpose and H_real.shape[0] == n_subcarriers:
        X = H_real.T
    else:
        raise ValueError(
            f"No reconozco el eje de subportadoras. Shape recibido: {H_real.shape}. "
            f"Esperaba (M,{n_subcarriers}) o ({n_subcarriers},M)."
        )

    return X[:, pilot_positions]


def extract_features_from_component_matrix(
    X_components: np.ndarray,
    *,
    fs: float = 1.0,
    bands: list[tuple[float, float]] | None = None,
    prefix: str = "comp",
    component_labels: list[str] | None = None,
) -> dict[str, float]:
    """
    Extrae características de una matriz genérica de componentes.

    Sirve tanto para:
        - componentes PCA
        - subportadoras piloto

    Entrada:
        X_components.shape = (M, C)

    Devuelve:
        diccionario plano de características.
    """
    X_components = np.asarray(X_components, dtype=np.float64)

    if X_components.ndim != 2:
        raise ValueError(
            f"X_components debe ser 2D, recibido shape={X_components.shape}"
        )

    _, C = X_components.shape

    if component_labels is None:
        component_labels = [f"{i:02d}" for i in range(C)]

    if len(component_labels) != C:
        raise ValueError(
            f"component_labels debe tener longitud {C}, "
            f"recibido {len(component_labels)}"
        )

    all_features: dict[str, float] = {}

    for c in range(C):
        x = X_components[:, c]

        feats_c = extract_signal_features(
            x,
            fs=fs,
            bands=bands,
        )

        label = component_labels[c]

        for key, value in feats_c.items():
            all_features[f"{prefix}_{label}_{key}"] = float(value)

    return all_features


# =============================================================================
# Función maestra
# =============================================================================

def extract_all_features_from_iq_bins(
    tx_bin_path: str | Path,
    rx_bin_path: str | Path,
    yaml_path: str | Path,
    *,
    n_symbols: int | None = None,
    start_sample_tx: int = 0,
    start_sample_rx: int = 0,
    tx_storage_format: IQStorageFormat = "auto",
    rx_storage_format: IQStorageFormat = "auto",
    fftshift: bool = False,
    normalize_fft: bool = False,
    n_components: int = 16,
    reduction_mode: ReductionMode = "pca",
    branches: tuple[FeatureBranch, ...] = ("logmag", "dphi"),
    dphi_keep_length: bool = False,
    symbol_stride: int = 1,
    bands: list[tuple[float, float]] | None = None,
    eps_channel: float = 1e-12,
    eps_log: float = 1e-12,
    return_intermediate: bool = False,
    verbose: bool = True,
) -> dict[str, float] | dict[str, Any]:
    """
    Función maestra para obtener all_features desde archivos IQ TX/RX.

    Flujo:
        iq_tx.bin, iq_rx.bin, Modulator.yaml
            -> X[m,k], Y[m,k]
            -> H[m,k] = Y[m,k] / X[m,k]
            -> log|H| y/o dphi
            -> reducción dimensional:
                   PCA individual, o selección de pilotos
            -> extracción de características
            -> all_features

    Parámetros principales
    ----------------------
    reduction_mode:
        "pca":
            Aplica PCA individual a cada captura.

        "pilots":
            No aplica PCA. Selecciona directamente las subportadoras
            indicadas en pilot_positions del YAML.

    branches:
        ("logmag", "dphi"):
            Extrae características de log|H| y de la diferencia temporal de fase.

    dphi_keep_length:
        False:
            dphi tiene M-1 muestras.

        True:
            dphi conserva M muestras rellenando la primera fila con ceros.

    bands:
        Bandas frecuenciales para las características espectrales.
        Si bands=None, se usan bandas relativas automáticas.

    return_intermediate:
        False:
            devuelve solo all_features.

        True:
            devuelve también X, Y, H, matrices reducidas, etc.
    """

    # ============================================================
    # 1. Leer configuración YAML
    # ============================================================

    cfg = load_modulator_yaml(yaml_path)

    fft_size = int(cfg["fft_size"])

    if n_symbols is None:
        n_symbols = int(cfg["num_symbols"])

    if reduction_mode not in ("pca", "pilots"):
        raise ValueError("reduction_mode debe ser 'pca' o 'pilots'.")

    if reduction_mode == "pilots" and fftshift:
        raise ValueError(
            "Para reduction_mode='pilots' usa fftshift=False, "
            "porque pilot_positions está definido en el orden natural de bins FFT."
        )

    if verbose:
        print("Configuración cargada")
        print(f"  fft_size: {fft_size}")
        print(f"  n_symbols usado: {n_symbols}")
        print(f"  reduction_mode: {reduction_mode}")
        print(f"  n_components PCA: {n_components}")
        print(f"  branches: {branches}")

    # ============================================================
    # 2. Leer IQ TX/RX
    # ============================================================

    tx_iq = read_usrp_iq_bin(
        tx_bin_path,
        cfg,
        storage_format=tx_storage_format,
        trim_to_complete_frames=False,
        return_time_axis=False,
        verbose=verbose,
    )

    rx_iq = read_usrp_iq_bin(
        rx_bin_path,
        cfg,
        storage_format=rx_storage_format,
        trim_to_complete_frames=False,
        return_time_axis=False,
        verbose=verbose,
    )

    # ============================================================
    # 3. Demodulación OFDM
    # ============================================================

    X = demodulate_ofdm_iq(
        tx_iq,
        cfg,
        n_symbols=n_symbols,
        start_sample=start_sample_tx,
        fftshift=fftshift,
        normalize_fft=normalize_fft,
        return_axes=False,
        verbose=verbose,
    )

    Y = demodulate_ofdm_iq(
        rx_iq,
        cfg,
        n_symbols=n_symbols,
        start_sample=start_sample_rx,
        fftshift=fftshift,
        normalize_fft=normalize_fft,
        return_axes=False,
        verbose=verbose,
    )

    # Recorte a longitud común por seguridad
    M_common = min(X.shape[0], Y.shape[0])

    if M_common <= 0:
        raise ValueError("No hay símbolos OFDM comunes entre TX y RX.")

    X = X[:M_common, :]
    Y = Y[:M_common, :]

    if X.shape[1] != fft_size or Y.shape[1] != fft_size:
        raise ValueError(
            f"Las rejillas no tienen {fft_size} subportadoras. "
            f"X={X.shape}, Y={Y.shape}"
        )

    # ============================================================
    # 4. Estimación de canal H = Y/X
    # ============================================================

    H, valid_mask = estimate_channel_grid(
        X,
        Y,
        eps=eps_channel,
        return_mask=True,
        verbose=verbose,
    )

    # ============================================================
    # 5. Diezmado temporal opcional
    # ============================================================

    if symbol_stride > 1:
        H = H[::symbol_stride, :]
        valid_mask = valid_mask[::symbol_stride, :]

        if verbose:
            print("Diezmado temporal aplicado")
            print(f"  symbol_stride: {symbol_stride}")
            print(f"  Nueva shape H: {H.shape}")

    fs_slow = _get_slow_time_fs_from_cfg(
        cfg,
        symbol_stride=symbol_stride,
    )

    if verbose:
        print(f"Frecuencia slow-time usada: {fs_slow:.6f} Hz")

    # ============================================================
    # 6. Construcción de ramas reales
    # ============================================================

    branch_matrices: dict[str, np.ndarray] = {}

    if "logmag" in branches:
        branch_matrices["logmag"] = _compute_logmag_from_H(
            H,
            eps=eps_log,
        )

    if "dphi" in branches:
        branch_matrices["dphi"] = _compute_dphi_from_H(
            H,
            keep_length=dphi_keep_length,
        )

    if len(branch_matrices) == 0:
        raise ValueError("No se ha seleccionado ninguna rama válida.")

    # ============================================================
    # 7. Preparación de reducción dimensional
    # ============================================================

    pilot_positions = None

    if reduction_mode == "pilots":
        pilot_positions = _get_pilot_positions_from_cfg(
            cfg,
            fft_size=fft_size,
        )

        if verbose:
            print("Modo de reducción: selección de pilotos")
            print(f"  pilot_positions: {pilot_positions.tolist()}")
            print(f"  Número de pilotos: {len(pilot_positions)}")

    elif reduction_mode == "pca":
        if verbose:
            print("Modo de reducción: PCA individual")

    # ============================================================
    # 8. Reducción + extracción de características
    # ============================================================

    all_features: dict[str, float] = {}

    reduced_outputs: dict[str, np.ndarray] = {}
    pca_models: dict[str, Any] = {}

    for branch_name, H_real in branch_matrices.items():

        if verbose:
            print(f"Procesando rama: {branch_name}")
            print(f"  Matriz real original: {H_real.shape}")

        if reduction_mode == "pca":

            if H_real.shape[0] < n_components:
                raise ValueError(
                    f"La rama {branch_name} tiene solo {H_real.shape[0]} muestras temporales. "
                    f"No puedes extraer {n_components} componentes PCA. "
                    f"Reduce n_components o usa más símbolos."
                )

            H_reduced, pca_model = pca_subcarriers_single_matrix(
                H_real,
                n_components=n_components,
                n_subcarriers=fft_size,
                auto_transpose=True,
            )

            component_labels = [f"{i:02d}" for i in range(H_reduced.shape[1])]

            pca_models[branch_name] = pca_model

            if verbose:
                print(f"  PCA: {H_real.shape} -> {H_reduced.shape}")

        elif reduction_mode == "pilots":

            H_reduced = _select_pilot_subcarriers_from_matrix(
                H_real,
                pilot_positions=pilot_positions,
                n_subcarriers=fft_size,
                auto_transpose=True,
            )

            component_labels = [
                f"pilot_{int(k):04d}" for k in pilot_positions
            ]

            if verbose:
                print(f"  Pilotos: {H_real.shape} -> {H_reduced.shape}")

        features_branch = extract_features_from_component_matrix(
            H_reduced,
            fs=fs_slow,
            bands=bands,
            prefix=branch_name,
            component_labels=component_labels,
        )

        all_features.update(features_branch)
        reduced_outputs[branch_name] = H_reduced

    # ============================================================
    # 9. Salida
    # ============================================================

    if not return_intermediate:
        return all_features

    return {
        "all_features": all_features,
        "cfg": cfg,
        "fs_slow": fs_slow,
        "X": X,
        "Y": Y,
        "H": H,
        "valid_mask": valid_mask,
        "branch_matrices": branch_matrices,
        "reduced_outputs": reduced_outputs,
        "pca_models": pca_models,
        "reduction_mode": reduction_mode,
        "pilot_positions": pilot_positions,
    }


def print_all_features(
    all_features: dict[str, float],
    *,
    sort_keys: bool = True,
    group_by_prefix: bool = True,
    decimals: int = 6,
    max_name_len: int = 55,
) -> None:
    """
    Imprime por pantalla todas las características contenidas en all_features.

    Entrada:
        all_features:
            Diccionario de características, por ejemplo el devuelto por
            extract_all_features_from_iq_bins(...)

    Ejemplo de clave:
        logmag_00_mean
        logmag_00_spectral_centroid
        dphi_15_energy_band_0_10_Hz
    """

    if not isinstance(all_features, dict):
        raise TypeError("all_features debe ser un diccionario.")

    if len(all_features) == 0:
        print("all_features está vacío.")
        return

    keys = list(all_features.keys())

    if sort_keys:
        keys = sorted(keys)

    print("=" * 80)
    print("CARACTERÍSTICAS EXTRAÍDAS")
    print("=" * 80)
    print(f"Número total de características: {len(keys)}")
    print("=" * 80)

    previous_group = None

    for key in keys:
        value = all_features[key]

        if group_by_prefix:
            # Espera nombres tipo:
            # logmag_00_mean
            # dphi_15_spectral_centroid
            parts = key.split("_")

            if len(parts) >= 2 and parts[1].isdigit():
                group = f"{parts[0]}_{parts[1]}"
            else:
                group = parts[0]

            if group != previous_group:
                print()
                print(f"[{group}]")
                print("-" * 80)
                previous_group = group

        if isinstance(value, (int, float)):
            value_str = f"{value:.{decimals}g}"
        else:
            value_str = str(value)

        name = key[:max_name_len]

        print(f"{name:<{max_name_len}} : {value_str}")