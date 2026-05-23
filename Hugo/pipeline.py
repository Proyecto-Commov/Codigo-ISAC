from __future__ import annotations
from pathlib import Path
from typing import Literal, Any
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.signal import stft

IQStorageFormat = Literal[
    "auto",
    "complex64",
    "sc16_interleaved",
    "sc8_interleaved",
]


def load_modulator_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """
    Lee el archivo Modulator.yaml y devuelve su contenido como diccionario.
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"No existe el archivo YAML: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("El YAML no contiene un diccionario válido.")

    return cfg


def _read_iq_with_format(
    bin_path: str | Path,
    fmt: Literal["complex64", "sc16_interleaved", "sc8_interleaved"],
) -> np.ndarray:
    """
    Lee un .bin de IQ según el formato indicado y devuelve un vector complejo.

    Salida:
        iq[n] = I[n] + j Q[n]
    """
    bin_path = Path(bin_path)

    if not bin_path.exists():
        raise FileNotFoundError(f"No existe el archivo binario: {bin_path}")

    if fmt == "complex64":
        return np.fromfile(bin_path, dtype=np.complex64)

    if fmt == "sc16_interleaved":
        raw = np.fromfile(bin_path, dtype=np.int16)

        if raw.size % 2 != 0:
            raise ValueError(
                "El archivo leído como sc16_interleaved tiene un número impar "
                "de enteros int16. No puede separarse en pares I/Q."
            )

        i = raw[0::2].astype(np.float32)
        q = raw[1::2].astype(np.float32)

        return (i + 1j * q).astype(np.complex64) / 32768.0

    if fmt == "sc8_interleaved":
        raw = np.fromfile(bin_path, dtype=np.int8)

        if raw.size % 2 != 0:
            raise ValueError(
                "El archivo leído como sc8_interleaved tiene un número impar "
                "de enteros int8. No puede separarse en pares I/Q."
            )

        i = raw[0::2].astype(np.float32)
        q = raw[1::2].astype(np.float32)

        return (i + 1j * q).astype(np.complex64) / 128.0

    raise ValueError(f"Formato IQ no soportado: {fmt}")


def _cp_tail_correlation_score(
    iq: np.ndarray,
    *,
    fft_size: int,
    cp_len: int,
    max_symbols: int = 50,
    eps: float = 1e-12,
) -> float:
    """
    Calcula una métrica sencilla de coherencia CP-cola.

    Si el formato de lectura es correcto, el prefijo cíclico debería parecerse
    bastante a la cola del símbolo OFDM útil.

    Devuelve un valor entre 0 y 1 aproximadamente.
    """
    iq = np.asarray(iq)

    if iq.ndim != 1:
        raise ValueError("La señal IQ debe ser un vector 1D.")

    sym_len = fft_size + cp_len
    n_symbols = len(iq) // sym_len

    if n_symbols <= 0:
        return 0.0

    n_symbols = min(n_symbols, max_symbols)

    x = iq[: n_symbols * sym_len].reshape(n_symbols, sym_len)

    cp = x[:, :cp_len]
    tail = x[:, -cp_len:]

    num = np.sum(cp * np.conj(tail), axis=1)
    den = np.sqrt(
        np.sum(np.abs(cp) ** 2, axis=1)
        * np.sum(np.abs(tail) ** 2, axis=1)
    ) + eps

    corr = np.abs(num / den)

    return float(np.nanmedian(corr))


def infer_iq_storage_format(
    bin_path: str | Path,
    yaml_path: str | Path,
) -> Literal["complex64", "sc16_interleaved", "sc8_interleaved"]:
    """
    Intenta inferir el formato real de almacenamiento del .bin.

    No se fía ciegamente de wire_format_tx/rx, porque eso describe el formato
    de transporte UHD, no necesariamente cómo se ha guardado el fichero.
    """
    cfg = load_modulator_yaml(yaml_path)

    fft_size = int(cfg["fft_size"])
    cp_len = int(cfg["cp_length"])

    candidates: list[Literal["complex64", "sc16_interleaved", "sc8_interleaved"]] = [
        "complex64",
        "sc16_interleaved",
        "sc8_interleaved",
    ]

    scores = {}

    for fmt in candidates:
        try:
            iq = _read_iq_with_format(bin_path, fmt)
            score = _cp_tail_correlation_score(
                iq,
                fft_size=fft_size,
                cp_len=cp_len,
            )
            scores[fmt] = score
        except Exception:
            scores[fmt] = -np.inf

    best_fmt = max(scores, key=scores.get)

    if not np.isfinite(scores[best_fmt]):
        raise ValueError("No se ha podido inferir ningún formato IQ válido.")

    return best_fmt


def read_usrp_iq_bin(
    bin_path: str | Path,
    yaml_path: str | Path,
    *,
    storage_format: IQStorageFormat = "auto",
    trim_to_complete_frames: bool = False,
    return_time_axis: bool = False,
    verbose: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Lee un archivo .bin de IQ USRP y devuelve la señal OFDM cruda en tiempo.

    Entrada:
        - bin_path: ruta al .bin, por ejemplo iq_rx.bin o iq_tx.bin
        - yaml_path: ruta al Modulator.yaml

    Salida:
        - iq: vector complejo 1D, sin quitar CP y sin FFT
        - t: eje temporal en segundos, si return_time_axis=True

    Notas:
        - Si storage_format="auto", se intenta inferir el formato real del .bin.
        - Si trim_to_complete_frames=True, se recorta el vector para dejar solo
          un número entero de tramas OFDM según el YAML.
        - El eje temporal se calcula como t[n] = n / sample_rate.
    """
    cfg = load_modulator_yaml(yaml_path)

    fft_size = int(cfg["fft_size"])
    cp_len = int(cfg["cp_length"])
    num_symbols = int(cfg["num_symbols"])
    sample_rate = float(cfg["sample_rate"])

    samples_per_symbol = fft_size + cp_len
    samples_per_frame = samples_per_symbol * num_symbols

    if storage_format == "auto":
        fmt = infer_iq_storage_format(bin_path, yaml_path)
    else:
        fmt = storage_format

    iq = _read_iq_with_format(bin_path, fmt)

    if trim_to_complete_frames:
        n_complete_frames = len(iq) // samples_per_frame

        if n_complete_frames == 0:
            raise ValueError(
                "El archivo no contiene ni una trama completa según el YAML."
            )

        iq = iq[: n_complete_frames * samples_per_frame]

    t = np.arange(len(iq), dtype=np.float64) / sample_rate

    if verbose:
        n_complete_symbols = len(iq) // samples_per_symbol
        n_complete_frames = len(iq) // samples_per_frame
        duration = len(iq) / sample_rate

        print("Archivo IQ leído correctamente")
        print(f"  Ruta: {bin_path}")
        print(f"  Formato detectado/usado: {fmt}")
        print(f"  Muestras complejas: {len(iq)}")
        print(f"  Frecuencia de muestreo: {sample_rate:.3f} Hz")
        print(f"  Duración temporal: {duration:.9f} s")
        print(f"  Muestras por símbolo OFDM: {samples_per_symbol}")
        print(f"  Muestras por trama OFDM: {samples_per_frame}")
        print(f"  Símbolos completos contenidos: {n_complete_symbols}")
        print(f"  Tramas completas contenidas: {n_complete_frames}")
        print(f"  dtype salida: {iq.dtype}")

    if return_time_axis:
        return iq, t

    return iq


def demodulate_ofdm_iq(
    iq: np.ndarray,
    yaml_path: str | Path,
    *,
    n_symbols: int | None = None,
    start_sample: int = 0,
    fftshift: bool = False,
    normalize_fft: bool = False,
    return_axes: bool = False,
    verbose: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Demodula una señal OFDM cruda en tiempo mediante:
    
        CP removal + FFT

    Entrada
    -------
    iq : np.ndarray
        Vector complejo 1D con la señal OFDM cruda en tiempo.
        Es decir, salida de read_usrp_iq_bin(...).

    yaml_path : str | Path
        Ruta al Modulator.yaml.

    n_symbols : int | None
        Número de símbolos OFDM a demodular.
        Si None, demodula todos los símbolos completos disponibles.

    start_sample : int
        Muestra inicial desde la que empezar a partir el vector en símbolos OFDM.
        Por defecto 0.

    fftshift : bool
        Si False, mantiene el orden natural de bins FFT: 0, 1, ..., N-1.
        Esto es lo más coherente con pilot_positions del YAML.
        Si True, centra DC en mitad del eje.

    normalize_fft : bool
        Si True, divide la FFT por sqrt(N).
        Si False, usa la convención por defecto de numpy.

    return_axes : bool
        Si True, devuelve también:
        - symbol_time_axis: eje temporal de símbolos OFDM [s]
        - subcarrier_axis: eje de subportadoras [Hz]

    Salida
    ------
    grid : np.ndarray
        Matriz OFDM compleja con shape (M, N).

        grid[m, k] = símbolo recibido/transmitido en:
            m = índice de símbolo OFDM
            k = índice de subportadora

    symbol_time_axis : np.ndarray, opcional
        Eje temporal de símbolos OFDM [s].

    subcarrier_axis : np.ndarray, opcional
        Eje frecuencial de subportadoras [Hz].
    """
    cfg = load_modulator_yaml(yaml_path)

    fft_size = int(cfg["fft_size"])
    cp_len = int(cfg["cp_length"])
    sample_rate = float(cfg["sample_rate"])

    iq = np.asarray(iq)

    if iq.ndim != 1:
        raise ValueError(f"iq debe ser un vector 1D, recibido shape {iq.shape}")

    if not np.iscomplexobj(iq):
        raise TypeError("iq debe ser complejo.")

    if start_sample < 0:
        raise ValueError("start_sample no puede ser negativo.")

    if start_sample >= len(iq):
        raise ValueError("start_sample está fuera del vector iq.")

    iq_work = iq[start_sample:]

    samples_per_symbol = fft_size + cp_len
    total_complete_symbols = len(iq_work) // samples_per_symbol

    if total_complete_symbols <= 0:
        raise ValueError(
            "No hay suficientes muestras para extraer un símbolo OFDM completo."
        )

    if n_symbols is None:
        n_symbols_used = total_complete_symbols
    else:
        if n_symbols <= 0:
            raise ValueError("n_symbols debe ser positivo.")
        n_symbols_used = min(int(n_symbols), total_complete_symbols)

    n_used_samples = n_symbols_used * samples_per_symbol

    iq_used = iq_work[:n_used_samples]

    # Shape: (M, N + Ncp)
    ofdm_symbols_with_cp = iq_used.reshape(n_symbols_used, samples_per_symbol)

    # Eliminación del CP
    ofdm_symbols_no_cp = ofdm_symbols_with_cp[:, cp_len:]

    # FFT sobre el eje de muestras dentro del símbolo
    grid = np.fft.fft(ofdm_symbols_no_cp, n=fft_size, axis=1)

    if normalize_fft:
        grid = grid / np.sqrt(fft_size)

    if fftshift:
        grid = np.fft.fftshift(grid, axes=1)

    # Ejes
    symbol_duration = samples_per_symbol / sample_rate
    symbol_time_axis = (
        start_sample / sample_rate
        + np.arange(n_symbols_used, dtype=np.float64) * symbol_duration
    )

    subcarrier_axis = np.fft.fftfreq(fft_size, d=1.0 / sample_rate)

    if fftshift:
        subcarrier_axis = np.fft.fftshift(subcarrier_axis)

    if verbose:
        print("Demodulación OFDM completada")
        print(f"  FFT size: {fft_size}")
        print(f"  CP length: {cp_len}")
        print(f"  Sample rate: {sample_rate:.3f} Hz")
        print(f"  Muestras por símbolo: {samples_per_symbol}")
        print(f"  Símbolos completos disponibles: {total_complete_symbols}")
        print(f"  Símbolos demodulados: {n_symbols_used}")
        print(f"  Shape rejilla OFDM: {grid.shape}")
        print(f"  fftshift: {fftshift}")
        print(f"  normalize_fft: {normalize_fft}")

    if return_axes:
        return grid, symbol_time_axis, subcarrier_axis

    return grid


def estimate_channel_grid(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    eps: float = 1e-12,
    return_mask: bool = True,
    verbose: bool = True,
):
    """
    Estima el canal OFDM sobre toda la rejilla tiempo-frecuencia.

    Entrada
    -------
    X : np.ndarray
        Rejilla transmitida, shape (M, N).

    Y : np.ndarray
        Rejilla recibida, shape (M, N).

    eps : float
        Umbral para evitar divisiones por cero o por valores muy pequeños de X.

    return_mask : bool
        Si True, devuelve también una máscara booleana de posiciones válidas.

    Salida
    ------
    H : np.ndarray
        Estimación de canal, shape (M, N).

        H[m, k] = Y[m, k] / X[m, k]

    valid_mask : np.ndarray, opcional
        Máscara booleana, shape (M, N), indicando dónde |X| > eps.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim != 2:
        raise ValueError(f"X debe tener shape (M, N), recibido {X.shape}")

    if Y.ndim != 2:
        raise ValueError(f"Y debe tener shape (M, N), recibido {Y.shape}")

    if X.shape != Y.shape:
        raise ValueError(
            f"X e Y deben tener la misma shape. "
            f"Recibido X={X.shape}, Y={Y.shape}"
        )

    if not np.iscomplexobj(X):
        raise TypeError("X debe ser una matriz compleja.")

    if not np.iscomplexobj(Y):
        raise TypeError("Y debe ser una matriz compleja.")

    valid_mask = np.abs(X) > eps

    H = np.full(X.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    H[valid_mask] = Y[valid_mask] / X[valid_mask]

    if verbose:
        M, N = X.shape
        n_valid = np.sum(valid_mask)
        n_total = valid_mask.size

        print("Estimación de canal completada")
        print(f"  Shape X: {X.shape}")
        print(f"  Shape Y: {Y.shape}")
        print(f"  Shape H: {H.shape}")
        print(f"  Símbolos OFDM M: {M}")
        print(f"  Subportadoras N: {N}")
        print(f"  Posiciones válidas: {n_valid}/{n_total}")
        print(f"  Porcentaje válido: {100 * n_valid / n_total:.2f} %")

    if return_mask:
        return H, valid_mask

    return H


def plot_ofdm_grid(
    grid,
    t_sym,
    f_sub,
    *,
    title="Rejilla OFDM",
    magnitude_db=False,
    freq_unit="MHz",
    time_unit="ms",
    fftshift_already_applied=False,
):
    """
    Representa módulo y fase de una rejilla OFDM compleja.

    Parámetros
    ----------
    grid : np.ndarray
        Matriz compleja con shape (M, N):
        eje 0 = símbolos OFDM
        eje 1 = subportadoras

    t_sym : np.ndarray
        Eje temporal de símbolos, shape (M,), en segundos.

    f_sub : np.ndarray
        Eje frecuencial de subportadoras, shape (N,), en Hz.

    fftshift_already_applied : bool
        - False: aplica fftshift internamente sobre el eje de subportadoras.
        - True: no aplica fftshift, asume que grid y f_sub ya están desplazados.
    """

    grid = np.asarray(grid)
    t_sym = np.asarray(t_sym)
    f_sub = np.asarray(f_sub)

    if grid.ndim != 2:
        raise ValueError(f"grid debe tener shape (M, N), recibido {grid.shape}")

    M, N = grid.shape

    if len(t_sym) != M:
        raise ValueError(f"t_sym debe tener longitud {M}, recibido {len(t_sym)}")

    if len(f_sub) != N:
        raise ValueError(f"f_sub debe tener longitud {N}, recibido {len(f_sub)}")

    # ============================================================
    # Reordenación frecuencial
    # ============================================================

    if fftshift_already_applied:
        grid_plot = grid
        f_plot = f_sub
    else:
        grid_plot = np.fft.fftshift(grid, axes=1)
        f_plot = np.fft.fftshift(f_sub)

    # ============================================================
    # Unidades de frecuencia
    # ============================================================

    if freq_unit == "Hz":
        f_axis = f_plot
        f_label = "Frecuencia de subportadora [Hz]"
    elif freq_unit == "kHz":
        f_axis = f_plot / 1e3
        f_label = "Frecuencia de subportadora [kHz]"
    elif freq_unit == "MHz":
        f_axis = f_plot / 1e6
        f_label = "Frecuencia de subportadora [MHz]"
    else:
        raise ValueError("freq_unit debe ser 'Hz', 'kHz' o 'MHz'.")

    # ============================================================
    # Unidades de tiempo
    # ============================================================

    if time_unit == "s":
        t_axis = t_sym
        t_label = "Tiempo [s]"
    elif time_unit == "ms":
        t_axis = t_sym * 1e3
        t_label = "Tiempo [ms]"
    elif time_unit == "us":
        t_axis = t_sym * 1e6
        t_label = "Tiempo [µs]"
    else:
        raise ValueError("time_unit debe ser 's', 'ms' o 'us'.")

    extent = [
        f_axis[0],
        f_axis[-1],
        t_axis[0],
        t_axis[-1],
    ]

    # ============================================================
    # Módulo
    # ============================================================

    mag = np.abs(grid_plot)

    if magnitude_db:
        mag = 20 * np.log10(np.maximum(mag, 1e-12))
        mag_label = "Módulo [dB]"
        mag_title = r"$20\log_{10}(|\cdot|)$"
    else:
        mag_label = "Módulo"
        mag_title = r"$|\cdot|$"

    # ============================================================
    # Fase
    # ============================================================

    phase = np.angle(grid_plot)

    # ============================================================
    # Figura
    # ============================================================

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    im0 = axes[0].imshow(
        mag,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=extent,
    )

    axes[0].set_title(f"{title} - Módulo: {mag_title}")
    axes[0].set_ylabel(t_label)

    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.set_label(mag_label)

    im1 = axes[1].imshow(
        phase,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=extent,
        cmap="twilight_shifted",
        vmin=-np.pi,
        vmax=np.pi,
    )

    axes[1].set_title(f"{title} - Fase: ∠(·)")
    axes[1].set_xlabel(f_label)
    axes[1].set_ylabel(t_label)

    cbar1 = plt.colorbar(im1, ax=axes[1])
    cbar1.set_label("Fase [rad]")
    cbar1.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar1.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    plt.tight_layout()
    plt.show()


def delay_time_ifft_from_H(
    H: np.ndarray,
    *,
    modulator_cfg: dict,
    n_subcarriers: int | None = None,
    pilot_subcarriers: np.ndarray | None = None,
    range_fft_size: int | None = None,
    apply_ifftshift: bool = False,
    distance_mode: str = "monostatic",
    c: float = 299_792_458.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica IFFT sobre el eje de subportadoras y obtiene h[tau,m],
    devolviendo además el eje de retardo convertido a distancia.

    Convención de entrada:
        H.shape = (M, N) si H está en toda la rejilla
        H.shape = (M, P) si H solo está en pilotos

    Convención de salida:
        h_tau_m.shape = (N_tau, M)
        distance_axis.shape = (N_tau,)

    distance_axis:
        - si distance_mode="monostatic":
            R = c*tau/2
        - si distance_mode="bistatic":
            d_excess = c*tau
    """

    H = np.asarray(H)

    if H.ndim != 2:
        raise ValueError(f"H debe tener shape (M, N) o (M, P), recibido {H.shape}")

    if not np.iscomplexobj(H):
        raise TypeError("H debe ser complejo.")

    fft_size = int(modulator_cfg["fft_size"])
    sample_rate = float(modulator_cfg["sample_rate"])

    delta_f = sample_rate / fft_size

    M, K = H.shape

    # Caso 1: H ya está en toda la rejilla
    if pilot_subcarriers is None:
        H_grid = H
        N = K

    # Caso 2: H solo está definido en pilotos
    else:
        pilot_subcarriers = np.asarray(pilot_subcarriers, dtype=int).reshape(-1)

        if pilot_subcarriers.size != K:
            raise ValueError(
                f"H tiene {K} columnas, pero pilot_subcarriers tiene "
                f"{pilot_subcarriers.size} elementos."
            )

        if n_subcarriers is None:
            n_subcarriers = fft_size

        N = int(n_subcarriers)

        if np.any((pilot_subcarriers < 0) | (pilot_subcarriers >= N)):
            raise IndexError("Hay subportadoras piloto fuera de rango.")

        H_grid = np.zeros((M, N), dtype=np.complex128)
        H_grid[:, pilot_subcarriers] = H

    if range_fft_size is None:
        range_fft_size = N

    if apply_ifftshift:
        H_grid = np.fft.ifftshift(H_grid, axes=1)

    # IFFT sobre subportadoras
    h_m_tau = np.fft.ifft(H_grid, n=range_fft_size, axis=1)

    # Devuelvo como h[tau,m]
    h_tau_m = h_m_tau.T

    tau_bins = np.arange(range_fft_size)

    tau_axis = tau_bins / (range_fft_size * delta_f)

    if distance_mode == "monostatic":
        distance_axis = c * tau_axis / 2.0

    elif distance_mode == "bistatic":
        distance_axis = c * tau_axis

    else:
        raise ValueError("distance_mode debe ser 'monostatic' o 'bistatic'.")

    return h_tau_m, distance_axis


def compute_fs_slow_from_cfg(cfg: dict) -> float:
    sample_rate = float(cfg["sample_rate"])
    fft_size = int(cfg["fft_size"])
    cp_len = int(cfg["cp_length"])
    stride = int(cfg.get("sensing_symbol_stride", 1))

    return sample_rate / (stride * (fft_size + cp_len))


def compute_range_axis_from_cfg(
    cfg: dict,
    n_range_bins: int,
    *,
    distance_mode: str = "monostatic",
    c: float = 299_792_458.0,
) -> np.ndarray:
    """
    Calcula el eje de rango/retardo en metros a partir de Modulator.yaml.
    """
    sample_rate = float(cfg["sample_rate"])
    fft_size = int(cfg["fft_size"])

    delta_f = sample_rate / fft_size

    tau_bins = np.arange(n_range_bins)
    tau_axis = tau_bins / (n_range_bins * delta_f)

    if distance_mode == "monostatic":
        return c * tau_axis / 2.0

    if distance_mode == "bistatic":
        return c * tau_axis

    raise ValueError("distance_mode debe ser 'monostatic' o 'bistatic'.")


def range_doppler_from_delay_time(
    h_tau_m: np.ndarray,
    *,
    modulator_cfg: dict,
    range_axis: np.ndarray | None = None,
    doppler_fft_size: int | None = None,
    window_slowtime: str | None = "hann",
    fftshift_doppler: bool = True,
    to_db: bool = False,
    power_floor: float = 1e-12,
    distance_mode: str = "monostatic",
    c: float = 299_792_458.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula el mapa rango-Doppler a partir de h[tau,m].

    Devuelve:
        RD_power : shape (N_tau, N_doppler)
        range_axis_out : eje de rango en metros
        doppler_axis : eje Doppler en Hz
    """

    h_tau_m = np.asarray(h_tau_m)

    if h_tau_m.ndim != 2:
        raise ValueError(f"h_tau_m debe tener shape (N_tau, M), recibido {h_tau_m.shape}")

    if not np.iscomplexobj(h_tau_m):
        raise TypeError("h_tau_m debe ser complejo.")

    N_tau, M = h_tau_m.shape

    if range_axis is None:
        range_axis_out = compute_range_axis_from_cfg(
            modulator_cfg,
            n_range_bins=N_tau,
            distance_mode=distance_mode,
            c=c,
        )
    else:
        range_axis_out = np.asarray(range_axis)
        if range_axis_out.ndim != 1 or range_axis_out.shape[0] != N_tau:
            raise ValueError(
                f"range_axis debe tener shape ({N_tau},), recibido {range_axis_out.shape}"
            )

    if doppler_fft_size is None:
        doppler_fft_size = M

    if doppler_fft_size < M:
        raise ValueError("doppler_fft_size debe ser >= M.")

    fs_slow = compute_fs_slow_from_cfg(modulator_cfg)

    if window_slowtime is None or window_slowtime == "rect":
        w = np.ones(M)
    elif window_slowtime == "hann":
        w = np.hanning(M)
    elif window_slowtime == "hamming":
        w = np.hamming(M)
    else:
        raise ValueError("window_slowtime debe ser None, 'rect', 'hann' o 'hamming'.")

    h_win = h_tau_m * w.reshape(1, M)

    RD_complex = np.fft.fft(
        h_win,
        n=doppler_fft_size,
        axis=1,
    )

    if fftshift_doppler:
        RD_complex = np.fft.fftshift(RD_complex, axes=1)

    RD_power = np.abs(RD_complex) ** 2

    if to_db:
        RD_power = 10.0 * np.log10(np.maximum(RD_power, power_floor))

    doppler_axis = np.fft.fftfreq(
        doppler_fft_size,
        d=1.0 / fs_slow,
    )

    if fftshift_doppler:
        doppler_axis = np.fft.fftshift(doppler_axis)

    return RD_power, range_axis_out, doppler_axis


def microdoppler_from_delay_time(
    h_tau_m: np.ndarray,
    *,
    modulator_cfg: dict,
    delay_bin: int | None = None,
    select_bin: str = "max_energy",
    nperseg: int = 64,
    noverlap: int = 48,
    nfft: int | None = None,
    window: str = "hann",
    fftshift_doppler: bool = True,
    to_db: bool = True,
    power_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Rama micro-Doppler tipo OpenISAC.

    Selecciona un bin de retardo y calcula la STFT sobre su señal slow-time.

    Parámetros
    ----------
    h_tau_m : np.ndarray
        Matriz retardo-tiempo.
        Shape: (N_tau, M)

        - eje 0: bins de retardo/rango
        - eje 1: símbolos OFDM / slow-time

    modulator_cfg : dict
        Diccionario leído desde Modulator.yaml mediante load_modulator_yaml(...).
        Se usa para calcular fs_slow.

    delay_bin : int | None
        Bin de retardo seleccionado manualmente.
        Si None, se selecciona automáticamente según select_bin.

    select_bin : str
        Método de selección automática del bin.
        Actualmente:
        - "max_energy": escoge el bin con mayor energía media.

    nperseg : int
        Longitud de ventana STFT.

    noverlap : int
        Solape entre ventanas STFT.

    nfft : int | None
        Tamaño de FFT de cada ventana STFT.
        Si None, se usa nperseg.

    window : str
        Ventana STFT, por ejemplo "hann" o "hamming".

    fftshift_doppler : bool
        Si True, centra cero Doppler.

    to_db : bool
        Si True, devuelve el espectrograma en dB.

    power_floor : float
        Suelo numérico para evitar log(0).

    Returns
    -------
    Smd : np.ndarray
        Espectrograma micro-Doppler.
        Shape: (N_f, N_t)

    f_md : np.ndarray
        Eje Doppler [Hz].
        Shape: (N_f,)

    t_md : np.ndarray
        Eje temporal [s].
        Shape: (N_t,)

    selected_delay_bin : int
        Bin de retardo usado.

    slow_signal : np.ndarray
        Señal slow-time compleja usada para calcular la STFT.
        Shape: (M,)
    """

    h_tau_m = np.asarray(h_tau_m)

    if h_tau_m.ndim != 2:
        raise ValueError(f"h_tau_m debe tener shape (N_tau, M), recibido {h_tau_m.shape}")

    if not np.iscomplexobj(h_tau_m):
        raise TypeError("h_tau_m debe ser complejo.")

    N_tau, M = h_tau_m.shape

    if M < 2:
        raise ValueError("Se necesitan al menos dos muestras slow-time.")

    fs_slow = compute_fs_slow_from_cfg(modulator_cfg)

    # ------------------------------------------------------------
    # Selección del bin de retardo
    # ------------------------------------------------------------
    if delay_bin is None:
        if select_bin == "max_energy":
            energy_per_bin = np.mean(np.abs(h_tau_m) ** 2, axis=1)
            delay_bin = int(np.argmax(energy_per_bin))
        else:
            raise ValueError("select_bin solo admite 'max_energy'.")

    if not (0 <= delay_bin < N_tau):
        raise IndexError(f"delay_bin={delay_bin} fuera de rango [0, {N_tau - 1}].")

    slow_signal = h_tau_m[delay_bin, :]

    # ------------------------------------------------------------
    # Parámetros STFT
    # ------------------------------------------------------------
    if nfft is None:
        nfft = nperseg

    if nperseg > M:
        raise ValueError(
            f"nperseg={nperseg} no puede ser mayor que la longitud slow-time M={M}."
        )

    if nfft < nperseg:
        raise ValueError("nfft debe ser >= nperseg.")

    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap debe cumplir 0 <= noverlap < nperseg.")

    # ------------------------------------------------------------
    # STFT
    # ------------------------------------------------------------
    f_md, t_md, Zxx = stft(
        slow_signal,
        fs=fs_slow,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    Smd = np.abs(Zxx) ** 2

    if fftshift_doppler:
        Smd = np.fft.fftshift(Smd, axes=0)
        f_md = np.fft.fftshift(f_md)

    if to_db:
        Smd = 10.0 * np.log10(np.maximum(Smd, power_floor))

    return Smd, f_md, t_md, delay_bin, slow_signal


def plot_rd_and_microdoppler(
    RD: np.ndarray,
    range_axis: np.ndarray,
    doppler_axis: np.ndarray,
    Smd: np.ndarray,
    f_md: np.ndarray,
    t_md: np.ndarray,
    *,
    delay_bin: int | None = None,
    rd_in_db: bool = True,
    microdoppler_in_db: bool = True,
    range_limits: tuple[float, float] | None = None,
    doppler_limits: tuple[float, float] | None = None,
    md_doppler_limits: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (14, 10),
    cmap_rd: str = "viridis",
    cmap_md: str = "viridis",
    aspect_rd: str = "auto",
    aspect_md: str = "auto",
    show_colorbars: bool = True,
    title: str | None = None,
):
    """
    Representa simultáneamente:
        1) mapa rango-Doppler
        2) espectrograma micro-Doppler

    Además, si se proporciona delay_bin, marca el bin dominante en el
    mapa rango-Doppler y muestra su distancia en metros.

    Parámetros
    ----------
    RD : np.ndarray
        Mapa rango-Doppler.
        Shape: (N_range, N_doppler)

    range_axis : np.ndarray
        Eje de rango en metros.
        Shape: (N_range,)

    doppler_axis : np.ndarray
        Eje Doppler del mapa rango-Doppler en Hz.
        Shape: (N_doppler,)

    Smd : np.ndarray
        Espectrograma micro-Doppler.
        Shape: (N_f, N_t)

    f_md : np.ndarray
        Eje Doppler del espectrograma micro-Doppler en Hz.
        Shape: (N_f,)

    t_md : np.ndarray
        Eje temporal del espectrograma en segundos.
        Shape: (N_t,)

    delay_bin : int | None
        Bin de retardo seleccionado para micro-Doppler.
        Si se proporciona, se marca en el mapa rango-Doppler.

    rd_in_db : bool
        True si RD ya está en dB.

    microdoppler_in_db : bool
        True si Smd ya está en dB.

    range_limits : tuple | None
        Límites del eje de rango en metros: (r_min, r_max)

    doppler_limits : tuple | None
        Límites Doppler del mapa rango-Doppler: (fd_min, fd_max)

    md_doppler_limits : tuple | None
        Límites Doppler del espectrograma micro-Doppler: (fd_min, fd_max)

    figsize : tuple
        Tamaño de figura.

    cmap_rd, cmap_md : str
        Mapas de color.

    show_colorbars : bool
        Si True, muestra barras de color.

    title : str | None
        Título global de la figura.
    """

    RD = np.asarray(RD)
    range_axis = np.asarray(range_axis)
    doppler_axis = np.asarray(doppler_axis)

    Smd = np.asarray(Smd)
    f_md = np.asarray(f_md)
    t_md = np.asarray(t_md)

    if RD.ndim != 2:
        raise ValueError(f"RD debe tener shape (N_range, N_doppler), recibido {RD.shape}")

    if Smd.ndim != 2:
        raise ValueError(f"Smd debe tener shape (N_f, N_t), recibido {Smd.shape}")

    if range_axis.ndim != 1 or range_axis.shape[0] != RD.shape[0]:
        raise ValueError(
            f"range_axis debe tener shape ({RD.shape[0]},), recibido {range_axis.shape}"
        )

    if doppler_axis.ndim != 1 or doppler_axis.shape[0] != RD.shape[1]:
        raise ValueError(
            f"doppler_axis debe tener shape ({RD.shape[1]},), recibido {doppler_axis.shape}"
        )

    if f_md.ndim != 1 or f_md.shape[0] != Smd.shape[0]:
        raise ValueError(
            f"f_md debe tener shape ({Smd.shape[0]},), recibido {f_md.shape}"
        )

    if t_md.ndim != 1 or t_md.shape[0] != Smd.shape[1]:
        raise ValueError(
            f"t_md debe tener shape ({Smd.shape[1]},), recibido {t_md.shape}"
        )

    dominant_distance = None

    if delay_bin is not None:
        if not (0 <= delay_bin < len(range_axis)):
            raise IndexError(
                f"delay_bin={delay_bin} fuera de rango [0, {len(range_axis)-1}]"
            )
        dominant_distance = float(range_axis[delay_bin])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        constrained_layout=True,
    )

    # ============================================================
    # MAPA RANGO-DOPPLER
    # ============================================================

    ax0 = axes[0]

    rd_extent = [
        doppler_axis[0],
        doppler_axis[-1],
        range_axis[0],
        range_axis[-1],
    ]

    im0 = ax0.imshow(
        RD,
        origin="lower",
        aspect=aspect_rd,
        extent=rd_extent,
        cmap=cmap_rd,
    )

    ax0.set_title("Mapa rango-Doppler")
    ax0.set_xlabel("Frecuencia Doppler [Hz]")
    ax0.set_ylabel("Rango [m]")

    if delay_bin is not None:
        ax0.axhline(
            dominant_distance,
            linestyle="--",
            linewidth=1.5,
            color="white",
            label=f"Bin dominante = {delay_bin}, R = {dominant_distance:.3f} m",
        )
        ax0.legend(loc="upper right")

    if doppler_limits is not None:
        ax0.set_xlim(*doppler_limits)

    if range_limits is not None:
        ax0.set_ylim(*range_limits)

    if show_colorbars:
        cbar0 = fig.colorbar(im0, ax=ax0)
        cbar0.set_label("Potencia [dB]" if rd_in_db else "Potencia")

    # ============================================================
    # ESPECTROGRAMA MICRO-DOPPLER
    # ============================================================

    ax1 = axes[1]

    md_extent_plot = [
        t_md[0],
        t_md[-1],
        f_md[0],
        f_md[-1],
    ]

    im1 = ax1.imshow(
        Smd,
        origin="lower",
        aspect=aspect_md,
        extent=md_extent_plot,
        cmap=cmap_md,
    )

    if delay_bin is not None:
        ax1.set_title(
            f"Espectrograma micro-Doppler "
            f"(bin dominante = {delay_bin}, R = {dominant_distance:.3f} m)"
        )
    else:
        ax1.set_title("Espectrograma micro-Doppler")

    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Frecuencia Doppler [Hz]")

    if md_doppler_limits is not None:
        ax1.set_ylim(*md_doppler_limits)

    if show_colorbars:
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label("Potencia [dB]" if microdoppler_in_db else "Potencia")

    if title is not None:
        fig.suptitle(title)

    plt.show()

    return fig, axes


def compute_fs_slow_from_cfg(
    cfg: dict,
) -> float:
    """
    Calcula la frecuencia de muestreo slow-time a partir
    del diccionario de configuración de Modulator.yaml.
    """

    sample_rate = float(cfg["sample_rate"])
    fft_size = int(cfg["fft_size"])
    cp_len = int(cfg["cp_length"])

    stride = int(cfg.get("sensing_symbol_stride", 1))

    fs_slow = sample_rate / (stride * (fft_size + cp_len))

    return fs_slow