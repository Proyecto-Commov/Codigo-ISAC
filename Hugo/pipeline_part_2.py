import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


# =============================================================================
# Análisis de componentes principales (PCA) para reducción de dimensionalidad
# =============================================================================

def pca_subcarriers_single_matrix(
    H_real: np.ndarray,
    n_components: int = 16,
    n_subcarriers: int = 1024,
    auto_transpose: bool = True,
):
    """
    Reduce una matriz real H_real de shape (M, N) a (M, n_components),
    aplicando PCA sobre el eje de subportadoras.

    Entrada esperada:
        H_real.shape = (M, N)
        M = número de símbolos OFDM / muestras slow-time
        N = número de subportadoras

    Para vuestro caso:
        H_real.shape = (100, 1024)

    Salida:
        H_pca.shape = (100, 16)

    Devuelve:
        H_pca, modelo_pca
    """

    H_real = np.asarray(H_real, dtype=np.float64)

    if H_real.ndim != 2:
        raise ValueError(f"H_real debe ser 2D, recibido shape={H_real.shape}")

    # En pipeline_rewritten.py lo normal es (M, N) = (símbolos, subportadoras).
    # Si llega como (1024, 100), lo transponemos.
    if H_real.shape[1] == n_subcarriers:
        X = H_real
    elif auto_transpose and H_real.shape[0] == n_subcarriers:
        X = H_real.T
    else:
        raise ValueError(
            f"No reconozco el eje de subportadoras. Shape recibido: {H_real.shape}. "
            f"Esperaba (M,{n_subcarriers}) o ({n_subcarriers},M)."
        )

    M, N = X.shape

    if n_components > min(M, N):
        raise ValueError(
            f"n_components={n_components} no puede ser mayor que min(M,N)={min(M,N)}"
        )

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=n_components))
    ])

    H_pca = model.fit_transform(X)

    return H_pca, model


def fit_pca_on_training_matrices(
    H_train_list: list[np.ndarray],
    n_components: int = 16,
    n_subcarriers: int = 1024,
):
    """
    Ajusta un PCA global usando únicamente las capturas de entrenamiento.

    Cada matriz debe tener shape:
        (M, N) = (símbolos, subportadoras)

    Por ejemplo:
        (100, 1024)

    Devuelve:
        modelo PCA global.
    """

    X_all = []

    for H in H_train_list:
        H = np.asarray(H, dtype=np.float64)

        if H.ndim != 2:
            raise ValueError(f"Cada H debe ser 2D, recibido {H.shape}")

        if H.shape[1] == n_subcarriers:
            X = H
        elif H.shape[0] == n_subcarriers:
            X = H.T
        else:
            raise ValueError(
                f"No reconozco el eje de subportadoras en shape={H.shape}"
            )

        X_all.append(X)

    # Apilamos todas las muestras slow-time de todas las capturas:
    # si hay C capturas de 100x1024 -> matriz total (C*100, 1024)
    X_train_pca = np.vstack(X_all)

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=n_components))
    ])

    model.fit(X_train_pca)

    return model


def transform_matrix_with_pca(
    H_real: np.ndarray,
    model,
    n_subcarriers: int = 1024,
):
    """
    Aplica un PCA ya ajustado a una matriz individual.

    Entrada:
        H_real.shape = (100, 1024)

    Salida:
        H_pca.shape = (100, 16)
    """

    H_real = np.asarray(H_real, dtype=np.float64)

    if H_real.shape[1] == n_subcarriers:
        X = H_real
    elif H_real.shape[0] == n_subcarriers:
        X = H_real.T
    else:
        raise ValueError(
            f"No reconozco el eje de subportadoras en shape={H_real.shape}"
        )

    H_pca = model.transform(X)

    return H_pca


# =============================================================================
# Extracción de características temporales y espectrales de señales 1D
# =============================================================================

def extract_signal_features(
    x,
    fs: float = 1.0,
    bands: list[tuple[float, float]] | None = None,
    autocorr_lags: tuple[int, ...] = (1, 2, 5),
    rolloff_percent: float = 0.85,
    use_hann_window: bool = True,
    remove_dc_for_spectral: bool = True,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Extrae características temporales y espectrales de una señal 1D.

    Parámetros
    ----------
    x:
        Señal real de entrada. Típicamente shape (100,) o (99,).
    fs:
        Frecuencia de muestreo slow-time en Hz.
        Si no quieres trabajar en Hz, deja fs=1.0.
    bands:
        Bandas frecuenciales para calcular energía.
        Ejemplo:
            bands=[(0,10), (10,30), (30,70), (70,150)]
        Si bands=None, se usan bandas relativas al fs.
    autocorr_lags:
        Retardos para autocorrelación normalizada.
    rolloff_percent:
        Porcentaje de energía acumulada para el roll-off.
    use_hann_window:
        Si True, aplica ventana Hann antes de FFT.
    remove_dc_for_spectral:
        Si True, elimina la media antes de calcular características espectrales.

    Devuelve
    --------
    features:
        Diccionario con características escalares.
    """

    x = np.asarray(x, dtype=np.float64).reshape(-1)

    if x.ndim != 1:
        raise ValueError("x debe ser una señal 1D.")

    if len(x) < 8:
        raise ValueError("La señal es demasiado corta para extraer características.")

    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    N = len(x)

    features = {}

    # ============================================================
    # 1. Características temporales básicas
    # ============================================================

    mean = np.mean(x)
    std = np.std(x)
    var = np.var(x)
    rms = np.sqrt(np.mean(x**2))
    energy = np.sum(x**2)
    abs_mean = np.mean(np.abs(x))
    peak = np.max(np.abs(x))
    peak_to_rms = peak / (rms + eps)

    x_centered = x - mean

    if std > eps:
        skewness = np.mean((x_centered / std) ** 3)
        kurtosis = np.mean((x_centered / std) ** 4)
    else:
        skewness = 0.0
        kurtosis = 0.0

    features["mean"] = float(mean)
    features["std"] = float(std)
    features["var"] = float(var)
    features["rms"] = float(rms)
    features["energy"] = float(energy)
    features["abs_mean"] = float(abs_mean)
    features["peak"] = float(peak)
    features["peak_to_rms"] = float(peak_to_rms)
    features["skewness"] = float(skewness)
    features["kurtosis"] = float(kurtosis)

    # ============================================================
    # 2. Autocorrelación normalizada
    # ============================================================

    denom = np.sum(x_centered**2) + eps

    for lag in autocorr_lags:
        if lag < N:
            autocorr = np.sum(x_centered[:-lag] * x_centered[lag:]) / denom
        else:
            autocorr = 0.0

        features[f"autocorr_lag_{lag}"] = float(autocorr)

    # ============================================================
    # 3. Preparación espectral
    # ============================================================

    x_spec = x.copy()

    if remove_dc_for_spectral:
        x_spec = x_spec - np.mean(x_spec)

    if use_hann_window:
        w = np.hanning(N)
        x_spec = x_spec * w

    # Zero-padding para suavizar el eje frecuencial
    nfft = int(2 ** np.ceil(np.log2(max(256, N))))

    X = np.fft.rfft(x_spec, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    power = np.abs(X) ** 2

    # Evitamos que DC domine si remove_dc no ha sido suficiente
    if remove_dc_for_spectral and len(power) > 1:
        power[0] = 0.0

    total_power = np.sum(power) + eps

    # ============================================================
    # 4. Centroide espectral
    # ============================================================

    spectral_centroid = np.sum(freqs * power) / total_power

    features["spectral_centroid"] = float(spectral_centroid)

    # ============================================================
    # 5. Ancho de banda espectral
    # ============================================================

    spectral_bandwidth = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * power) / total_power
    )

    features["spectral_bandwidth"] = float(spectral_bandwidth)

    # ============================================================
    # 6. Roll-off espectral
    # ============================================================

    cumulative_power = np.cumsum(power)
    threshold = rolloff_percent * cumulative_power[-1]

    idx_rolloff = np.searchsorted(cumulative_power, threshold)

    if idx_rolloff >= len(freqs):
        idx_rolloff = len(freqs) - 1

    spectral_rolloff = freqs[idx_rolloff]

    features[f"spectral_rolloff_{int(rolloff_percent * 100)}"] = float(spectral_rolloff)

    # ============================================================
    # 7. Spectral flatness
    # ============================================================

    positive_power = power + eps

    spectral_flatness = np.exp(np.mean(np.log(positive_power))) / (
        np.mean(positive_power) + eps
    )

    features["spectral_flatness"] = float(spectral_flatness)

    # ============================================================
    # 8. Contraste espectral simple
    # ============================================================

    # Se calcula como diferencia en dB entre percentil alto y bajo del espectro.
    power_db = 10.0 * np.log10(power + eps)

    spectral_contrast = np.percentile(power_db, 95) - np.percentile(power_db, 5)

    features["spectral_contrast"] = float(spectral_contrast)

    # ============================================================
    # 9. Energía por bandas
    # ============================================================

    if bands is None:
        # Bandas relativas si no se da fs real.
        # Cubren desde 0 hasta Nyquist.
        fmax = fs / 2.0
        bands = [
            (0.00 * fmax, 0.10 * fmax),
            (0.10 * fmax, 0.25 * fmax),
            (0.25 * fmax, 0.50 * fmax),
            (0.50 * fmax, 1.00 * fmax),
        ]

    for f_low, f_high in bands:
        mask = (freqs >= f_low) & (freqs < f_high)

        band_power = np.sum(power[mask])
        band_power_ratio = band_power / total_power

        name = f"band_{f_low:g}_{f_high:g}_Hz"

        features[f"energy_{name}"] = float(band_power)
        features[f"energy_ratio_{name}"] = float(band_power_ratio)

    # ============================================================
    # 10. Frecuencia dominante
    # ============================================================

    idx_peak = int(np.argmax(power))
    dominant_freq = freqs[idx_peak]
    dominant_power_ratio = power[idx_peak] / total_power

    features["dominant_freq"] = float(dominant_freq)
    features["dominant_power_ratio"] = float(dominant_power_ratio)

    return features


def extract_features_from_pca_matrix(
    X_pca,
    fs: float = 1.0,
    bands: list[tuple[float, float]] | None = None,
    prefix: str = "pca",
) -> dict[str, float]:
    """
    Extrae características de una matriz PCA de shape (M, C).

    Ejemplo:
        X_pca.shape = (100, 16)

    Interpreta:
        M = muestras temporales
        C = componentes PCA

    Devuelve:
        diccionario plano con nombres tipo:
            pca_00_mean
            pca_00_std
            ...
            pca_15_spectral_centroid
    """

    X_pca = np.asarray(X_pca, dtype=np.float64)

    if X_pca.ndim != 2:
        raise ValueError(f"X_pca debe ser 2D, recibido shape={X_pca.shape}")

    M, C = X_pca.shape

    all_features = {}

    for c in range(C):
        x = X_pca[:, c]

        feats_c = extract_signal_features(
            x,
            fs=fs,
            bands=bands,
        )

        for key, value in feats_c.items():
            all_features[f"{prefix}_{c:02d}_{key}"] = value

    return all_features


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