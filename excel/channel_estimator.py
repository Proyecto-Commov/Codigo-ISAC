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

# =============================================================================
# Configuración YAML
# =============================================================================

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


# =============================================================================
# Lectura IQ y demodulación OFDM
# =============================================================================

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
    cfg: dict[str, Any],
) -> Literal["complex64", "sc16_interleaved", "sc8_interleaved"]:
    """
    Intenta inferir el formato real de almacenamiento del .bin.

    No se fía ciegamente de wire_format_tx/rx, porque eso describe el formato
    de transporte UHD, no necesariamente cómo se ha guardado el fichero.
    """
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
    cfg: dict[str, Any],
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
        - cfg: configuración del modulador OFDM

    Salida:
        - iq: vector complejo 1D, sin quitar CP y sin FFT
        - t: eje temporal en segundos, si return_time_axis=True

    Notas:
        - Si storage_format="auto", se intenta inferir el formato real del .bin.
        - Si trim_to_complete_frames=True, se recorta el vector para dejar solo
          un número entero de tramas OFDM según el YAML.
        - El eje temporal se calcula como t[n] = n / sample_rate.
    """
    fft_size = int(cfg["fft_size"])
    cp_len = int(cfg["cp_length"])
    num_symbols = int(cfg["num_symbols"])
    sample_rate = float(cfg["sample_rate"])

    samples_per_symbol = fft_size + cp_len
    samples_per_frame = samples_per_symbol * num_symbols

    if storage_format == "auto":
        fmt = infer_iq_storage_format(bin_path, cfg)
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
    cfg: dict[str, Any],
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

    cfg : dict[str, Any]
        Configuración del modulador OFDM.

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


# =============================================================================
# Estimación de canal
# =============================================================================

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


# =============================================================================
# Representación gráfica
# =============================================================================

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


# =============================================================================
# Función maestra: paths TX/RX/YAML -> matriz de canal H
# =============================================================================

def compute_channel_matrix_from_iq_paths(
    tx_path: str | Path,
    rx_path: str | Path,
    yaml_path: str | Path,
    *,
    n_symbols: int | None = None,
    start_sample_tx: int = 0,
    start_sample_rx: int = 0,
    storage_format_tx: IQStorageFormat = "auto",
    storage_format_rx: IQStorageFormat = "auto",
    fftshift: bool = False,
    normalize_fft: bool = False,
    eps: float = 1e-12,
    trim_to_complete_frames: bool = True,
    return_aux: bool = False,
    output_order: Literal["mk", "km"] = "mk",
    verbose: bool = True,
):
    """
    Función maestra para obtener la matriz de canal OFDM H a partir de:

        - tx_path: ruta a iq_tx.bin
        - rx_path: ruta a iq_rx.bin
        - yaml_path: ruta a Modulator.yaml

    Pipeline interno
    ----------------
    1. Lee el YAML.
    2. Lee la señal IQ transmitida.
    3. Lee la señal IQ recibida.
    4. Demodula ambas señales OFDM:
            CP removal + FFT
    5. Obtiene las rejillas:
            X[m, k] = símbolo transmitido
            Y[m, k] = símbolo recibido
    6. Estima el canal:
            H[m, k] = Y[m, k] / X[m, k]

    Parámetros importantes
    ----------------------
    n_symbols:
        Número de símbolos OFDM a procesar.
        Si None, se usan todos los símbolos completos comunes entre TX y RX.

    start_sample_tx:
        Muestra inicial para cortar la señal TX antes de demodular.

    start_sample_rx:
        Muestra inicial para cortar la señal RX antes de demodular.
        Si existe un desfase temporal conocido entre TX y RX, puede corregirse aquí.

    storage_format_tx, storage_format_rx:
        Formato real del .bin.
        Opciones:
            "auto"
            "complex64"
            "sc16_interleaved"
            "sc8_interleaved"

    fftshift:
        Si False, mantiene el orden natural FFT:
            k = 0, 1, ..., N-1
        Esto es lo recomendable si quieres que los índices coincidan con
        pilot_positions del YAML.

    normalize_fft:
        Si True, divide la FFT entre sqrt(N).
        Como se aplica tanto a TX como a RX, normalmente no cambia H,
        salvo efectos numéricos.

    output_order:
        - "mk": devuelve H con shape (M, N), es decir H[m, k].
                Es el formato usado internamente en este archivo.
        - "km": devuelve H con shape (N, M), es decir H[k, m].
                Es el formato más habitual en la notación teórica.

    return_aux:
        Si False:
            devuelve solo H.
        Si True:
            devuelve un diccionario con H, X, Y, máscara válida, ejes y cfg.

    Salida
    ------
    H:
        Matriz compleja de canal.

        Si output_order="mk":
            H.shape = (M, N)
            H[m, k]

        Si output_order="km":
            H.shape = (N, M)
            H[k, m]
    """

    # ------------------------------------------------------------
    # 1. Cargar configuración
    # ------------------------------------------------------------
    cfg = load_modulator_yaml(yaml_path)

    # ------------------------------------------------------------
    # 2. Leer IQ TX/RX
    # ------------------------------------------------------------
    iq_tx = read_usrp_iq_bin(
        tx_path,
        cfg,
        storage_format=storage_format_tx,
        trim_to_complete_frames=trim_to_complete_frames,
        return_time_axis=False,
        verbose=verbose,
    )

    iq_rx = read_usrp_iq_bin(
        rx_path,
        cfg,
        storage_format=storage_format_rx,
        trim_to_complete_frames=trim_to_complete_frames,
        return_time_axis=False,
        verbose=verbose,
    )

    # ------------------------------------------------------------
    # 3. Determinar número común de símbolos si no se especifica
    # ------------------------------------------------------------
    fft_size = int(cfg["fft_size"])
    cp_len = int(cfg["cp_length"])
    sample_rate = float(cfg["sample_rate"])

    samples_per_symbol = fft_size + cp_len

    tx_symbols_available = (len(iq_tx) - start_sample_tx) // samples_per_symbol
    rx_symbols_available = (len(iq_rx) - start_sample_rx) // samples_per_symbol

    if tx_symbols_available <= 0:
        raise ValueError("La señal TX no contiene símbolos OFDM completos útiles.")

    if rx_symbols_available <= 0:
        raise ValueError("La señal RX no contiene símbolos OFDM completos útiles.")

    max_common_symbols = min(tx_symbols_available, rx_symbols_available)

    if n_symbols is None:
        n_symbols_used = max_common_symbols
    else:
        if n_symbols <= 0:
            raise ValueError("n_symbols debe ser positivo.")
        n_symbols_used = min(int(n_symbols), max_common_symbols)

    if verbose:
        print("Selección de símbolos común TX/RX")
        print(f"  Símbolos disponibles TX: {tx_symbols_available}")
        print(f"  Símbolos disponibles RX: {rx_symbols_available}")
        print(f"  Símbolos usados: {n_symbols_used}")

    # ------------------------------------------------------------
    # 4. Demodular TX y RX: CP removal + FFT
    # ------------------------------------------------------------
    X, t_sym, f_sub = demodulate_ofdm_iq(
        iq_tx,
        cfg,
        n_symbols=n_symbols_used,
        start_sample=start_sample_tx,
        fftshift=fftshift,
        normalize_fft=normalize_fft,
        return_axes=True,
        verbose=verbose,
    )

    Y = demodulate_ofdm_iq(
        iq_rx,
        cfg,
        n_symbols=n_symbols_used,
        start_sample=start_sample_rx,
        fftshift=fftshift,
        normalize_fft=normalize_fft,
        return_axes=False,
        verbose=verbose,
    )

    # ------------------------------------------------------------
    # 5. Estimar canal H = Y / X
    # ------------------------------------------------------------
    H, valid_mask = estimate_channel_grid(
        X,
        Y,
        eps=eps,
        return_mask=True,
        verbose=verbose,
    )

    # ------------------------------------------------------------
    # 6. Eje slow-time efectivo
    # ------------------------------------------------------------
    symbol_duration = samples_per_symbol / sample_rate
    fs_slow = 1.0 / symbol_duration

    if verbose:
        print("Matriz de canal obtenida")
        print(f"  Shape H interna: {H.shape} = (símbolos, subportadoras)")
        print(f"  Duración símbolo OFDM con CP: {symbol_duration:.9e} s")
        print(f"  Frecuencia slow-time: {fs_slow:.3f} Hz")

    # ------------------------------------------------------------
    # 7. Orden de salida
    # ------------------------------------------------------------
    if output_order == "mk":
        H_out = H
        X_out = X
        Y_out = Y
        valid_mask_out = valid_mask
    elif output_order == "km":
        H_out = H.T
        X_out = X.T
        Y_out = Y.T
        valid_mask_out = valid_mask.T
    else:
        raise ValueError("output_order debe ser 'mk' o 'km'.")

    if not return_aux:
        return H_out

    return {
        "H": H_out,
        "X": X_out,
        "Y": Y_out,
        "valid_mask": valid_mask_out,
        "t_sym": t_sym,
        "f_sub": f_sub,
        "fs_slow": fs_slow,
        "symbol_duration": symbol_duration,
        "cfg": cfg,
        "n_symbols_used": n_symbols_used,
        "output_order": output_order,
    }