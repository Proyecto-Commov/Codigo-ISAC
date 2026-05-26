from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

@dataclass
class PilotObservation:
    """
    Contenedor para la extracción de Y[k,m] solo en las posiciones de las subportadoras piloto.

    Convención:
    - rx_grid: shape (M, N), si se conserva
        M = número de símbolos OFDM
        N = número de subportadoras
    - y_pilots: shape (M_eff, P_eff)
        M_eff = número de símbolos seleccionados
        P_eff = número de subportadoras piloto seleccionadas

    Interpretación:
    - y_pilots[i, j] = Y[k_j, m_i]
    """
    source: Literal["sionna", "usrp_iq"]
    domain: Literal["freq", "mixed"]

    y_pilots: np.ndarray
    pilot_subcarriers: np.ndarray
    pilot_symbol_indices: np.ndarray

    rx_grid: np.ndarray | None = None
    samples_td: np.ndarray | None = None

    meta: dict[str, Any] = field(default_factory=dict)


def _ensure_complex_ndarray(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if not np.iscomplexobj(arr):
        raise TypeError(f"{name} debe ser complejo.")
    return arr


def _to_grid_mn(arr: np.ndarray) -> np.ndarray:
    """
    Convierte una entrada tipo Sionna a shape (M, N).

    Casos soportados:
    - (M, N)
    - (batch, M, N)
    - (batch, tx, streams, M, N)
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 5:
        return arr[0, 0, 0]

    raise ValueError(f"No puedo interpretar shape {arr.shape} como rejilla OFDM.")


def _remove_cp_and_fft(
    iq: np.ndarray,
    fft_size: int,
    cp_len: int,
    n_symbols: int | None = None,
) -> np.ndarray:
    """
    Convierte IQ temporal 1D en rejilla OFDM recibida rx_grid con shape (M, N).
    """
    if iq.ndim != 1:
        raise ValueError("La señal IQ temporal debe ser 1D.")

    sym_len = fft_size + cp_len
    total_symbols = len(iq) // sym_len

    if total_symbols <= 0:
        raise ValueError("No hay suficientes muestras para extraer símbolos OFDM.")

    if n_symbols is not None:
        total_symbols = min(total_symbols, n_symbols)

    trimmed = iq[: total_symbols * sym_len]
    frames = trimmed.reshape(total_symbols, sym_len)

    no_cp = frames[:, cp_len:]
    rx_grid = np.fft.fft(no_cp, axis=1)

    return rx_grid


def _read_usrp_iq_file(
    path: str | Path,
    dtype: Literal["complex64", "sc16_interleaved"] = "complex64",
) -> np.ndarray:
    """
    Lee archivo IQ crudo de USRP.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    if dtype == "complex64":
        return np.fromfile(path, dtype=np.complex64)

    if dtype == "sc16_interleaved":
        raw = np.fromfile(path, dtype=np.int16)
        if raw.size % 2 != 0:
            raise ValueError("Archivo IQ interleaved con número impar de enteros.")
        i = raw[0::2].astype(np.float32)
        q = raw[1::2].astype(np.float32)
        return (i + 1j * q) / 32768.0

    raise ValueError(f"dtype no soportado: {dtype}")


def _validate_pilot_spec(
    rx_grid: np.ndarray,
    pilot_subcarriers: np.ndarray,
    pilot_symbol_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Valida y normaliza la especificación de pilotos.
    """
    M, N = rx_grid.shape

    pilot_subcarriers = np.asarray(pilot_subcarriers, dtype=int).reshape(-1)
    if pilot_subcarriers.size == 0:
        raise ValueError("pilot_subcarriers no puede estar vacío.")
    if np.any((pilot_subcarriers < 0) | (pilot_subcarriers >= N)):
        raise IndexError(f"Hay subportadoras piloto fuera de rango [0, {N-1}].")

    if pilot_symbol_indices is None:
        pilot_symbol_indices = np.arange(M, dtype=int)
    else:
        pilot_symbol_indices = np.asarray(pilot_symbol_indices, dtype=int).reshape(-1)
        if pilot_symbol_indices.size == 0:
            raise ValueError("pilot_symbol_indices no puede estar vacío.")
        if np.any((pilot_symbol_indices < 0) | (pilot_symbol_indices >= M)):
            raise IndexError(f"Hay símbolos piloto fuera de rango [0, {M-1}].")

    return pilot_subcarriers, pilot_symbol_indices


def _extract_pilot_observations(
    rx_grid: np.ndarray,
    pilot_subcarriers: np.ndarray,
    pilot_symbol_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrae Y[k,m] solo en posiciones piloto.

    Returns
    -------
    y_pilots : np.ndarray
        shape (M_eff, P_eff)
    pilot_subcarriers : np.ndarray
    pilot_symbol_indices : np.ndarray
    """
    pilot_subcarriers, pilot_symbol_indices = _validate_pilot_spec(
        rx_grid, pilot_subcarriers, pilot_symbol_indices
    )

    # Selecciona primero símbolos, luego subportadoras
    y_pilots = rx_grid[np.ix_(pilot_symbol_indices, pilot_subcarriers)]

    return y_pilots, pilot_subcarriers, pilot_symbol_indices


def adapt_to_pilot_observations(
    source_type: Literal["sionna", "usrp_iq"],
    *,
    data: Any | None = None,
    file_path: str | Path | None = None,
    data_kind: Literal["grid", "time"] | None = None,
    pilot_subcarriers: np.ndarray,
    pilot_symbol_indices: np.ndarray | None = None,
    fft_size: int | None = None,
    cp_len: int | None = None,
    n_symbols: int | None = None,
    usrp_file_dtype: Literal["complex64", "sc16_interleaved"] = "complex64",
    keep_rx_grid: bool = False,
) -> PilotObservation:
    """
    Devuelve Y[k,m] únicamente en posiciones piloto.

    Parámetros
    ----------
    source_type : {"sionna", "usrp_iq"}

    Caso Sionna
    -----------
    data_kind="grid":
        `data` es una rejilla OFDM ya en frecuencia.
    data_kind="time":
        `data` es señal temporal compleja; requiere fft_size y cp_len.

    Caso USRP
    ---------
    file_path o data:
        IQ temporal crudo; requiere fft_size y cp_len.

    pilot_subcarriers : np.ndarray
        Vector de subportadoras piloto, p.ej. [0,4,8,...,28].

    pilot_symbol_indices : np.ndarray | None
        Símbolos OFDM donde quieres extraer esos pilotos.
        Si None, usa todos los símbolos disponibles.

    keep_rx_grid : bool
        Si True, conserva la rejilla completa rx_grid para depuración.

    Returns
    -------
    PilotObservation
        con y_pilots shape (M_eff, P_eff)
    """
    if source_type == "sionna":
        if data is None:
            raise ValueError("Para source_type='sionna' debes pasar `data`.")
        if data_kind is None:
            raise ValueError("Para Sionna debes indicar data_kind='grid' o 'time'.")

        arr = np.asarray(data)

        if data_kind == "grid":
            rx_grid = _to_grid_mn(_ensure_complex_ndarray(arr, "data"))
            y_pilots, pilot_subcarriers, pilot_symbol_indices = _extract_pilot_observations(
                rx_grid,
                pilot_subcarriers=pilot_subcarriers,
                pilot_symbol_indices=pilot_symbol_indices,
            )

            return PilotObservation(
                source="sionna",
                domain="freq",
                y_pilots=y_pilots,
                pilot_subcarriers=pilot_subcarriers,
                pilot_symbol_indices=pilot_symbol_indices,
                rx_grid=rx_grid if keep_rx_grid else None,
                meta={
                    "input_kind": "sionna_grid",
                    "shape_rx_grid": tuple(rx_grid.shape),
                    "shape_y_pilots": tuple(y_pilots.shape),
                },
            )

        if data_kind == "time":
            if fft_size is None or cp_len is None:
                raise ValueError("Para data_kind='time' debes pasar fft_size y cp_len.")

            samples_td = _ensure_complex_ndarray(arr, "data").reshape(-1)
            rx_grid = _remove_cp_and_fft(samples_td, fft_size, cp_len, n_symbols=n_symbols)

            y_pilots, pilot_subcarriers, pilot_symbol_indices = _extract_pilot_observations(
                rx_grid,
                pilot_subcarriers=pilot_subcarriers,
                pilot_symbol_indices=pilot_symbol_indices,
            )

            return PilotObservation(
                source="sionna",
                domain="mixed",
                y_pilots=y_pilots,
                pilot_subcarriers=pilot_subcarriers,
                pilot_symbol_indices=pilot_symbol_indices,
                rx_grid=rx_grid if keep_rx_grid else None,
                samples_td=samples_td,
                meta={
                    "input_kind": "sionna_time",
                    "fft_size": fft_size,
                    "cp_len": cp_len,
                    "shape_rx_grid": tuple(rx_grid.shape),
                    "shape_y_pilots": tuple(y_pilots.shape),
                },
            )

        raise ValueError(f"data_kind no soportado para Sionna: {data_kind}")

    if source_type == "usrp_iq":
        if fft_size is None or cp_len is None:
            raise ValueError("Para source_type='usrp_iq' debes pasar fft_size y cp_len.")

        if file_path is None and data is None:
            raise ValueError("Debes pasar `file_path` o `data` para USRP IQ.")

        if file_path is not None:
            samples_td = _read_usrp_iq_file(file_path, dtype=usrp_file_dtype)
        else:
            samples_td = _ensure_complex_ndarray(data, "data").reshape(-1)

        rx_grid = _remove_cp_and_fft(samples_td, fft_size, cp_len, n_symbols=n_symbols)

        y_pilots, pilot_subcarriers, pilot_symbol_indices = _extract_pilot_observations(
            rx_grid,
            pilot_subcarriers=pilot_subcarriers,
            pilot_symbol_indices=pilot_symbol_indices,
        )

        return PilotObservation(
            source="usrp_iq",
            domain="mixed",
            y_pilots=y_pilots,
            pilot_subcarriers=pilot_subcarriers,
            pilot_symbol_indices=pilot_symbol_indices,
            rx_grid=rx_grid if keep_rx_grid else None,
            samples_td=samples_td,
            meta={
                "input_kind": "usrp_iq",
                "fft_size": fft_size,
                "cp_len": cp_len,
                "shape_rx_grid": tuple(rx_grid.shape),
                "shape_y_pilots": tuple(y_pilots.shape),
                "usrp_file_dtype": usrp_file_dtype if file_path is not None else "array",
            },
        )

    raise ValueError(f"source_type no soportado: {source_type}")



def estimate_channel_on_pilots(
    y_pilots: np.ndarray,
    pilot_values: np.ndarray,
    eps: float = 1e-12,
    return_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Estima el canal H[k,m] únicamente en posiciones piloto.

    Parámetros
    ----------
    y_pilots : np.ndarray
        Matriz compleja con las observaciones recibidas en pilotos.
        Shape esperada: (M_eff, P_eff)

        - M_eff = número de símbolos OFDM seleccionados
        - P_eff = número de subportadoras piloto

    pilot_values : np.ndarray
        Valores complejos transmitidos conocidos en esos pilotos.

        Casos permitidos:
        - shape (P_eff,)      -> mismos pilotos para todos los símbolos
        - shape (M_eff, P_eff)-> pilotos que pueden cambiar con el símbolo

    eps : float
        Umbral para evitar divisiones por cero.

    return_mask : bool
        Si True, devuelve también una máscara booleana de posiciones válidas.

    Returns
    -------
    h_pilots : np.ndarray
        Estimación de canal en pilotos.
        Shape: (M_eff, P_eff)

    valid_mask : np.ndarray, opcional
        Máscara booleana de posiciones donde la estimación es válida.
    """
    y_pilots = np.asarray(y_pilots)
    pilot_values = np.asarray(pilot_values)

    if y_pilots.ndim != 2:
        raise ValueError(
            f"y_pilots debe tener shape (M_eff, P_eff), recibido {y_pilots.shape}"
        )

    if not np.iscomplexobj(y_pilots):
        raise TypeError("y_pilots debe ser complejo.")

    M_eff, P_eff = y_pilots.shape

    if pilot_values.ndim == 1:
        if pilot_values.shape[0] != P_eff:
            raise ValueError(
                f"pilot_values tiene longitud {pilot_values.shape[0]}, "
                f"pero P_eff = {P_eff}"
            )
        pilot_matrix = np.broadcast_to(pilot_values.reshape(1, P_eff), (M_eff, P_eff))

    elif pilot_values.ndim == 2:
        if pilot_values.shape != (M_eff, P_eff):
            raise ValueError(
                f"pilot_values debe tener shape {(M_eff, P_eff)}, "
                f"recibido {pilot_values.shape}"
            )
        pilot_matrix = pilot_values

    else:
        raise ValueError(
            "pilot_values debe tener shape (P_eff,) o (M_eff, P_eff)."
        )

    if not np.iscomplexobj(pilot_matrix):
        raise TypeError("pilot_values debe ser complejo.")

    valid_mask = np.abs(pilot_matrix) > eps

    h_pilots = np.full((M_eff, P_eff), np.nan + 1j * np.nan, dtype=np.complex128)
    h_pilots[valid_mask] = y_pilots[valid_mask] / pilot_matrix[valid_mask]

    if return_mask:
        return h_pilots, valid_mask

    return h_pilots



def phase_difference_on_pilots(
    h_pilots: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    use_conjugate_order: str = "current_next",
    return_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Calcula la diferencia de fase entre símbolos consecutivos sobre pilotos.

    Parámetros
    ----------
    h_pilots : np.ndarray
        Canal estimado en pilotos.
        Shape esperada: (M_eff, P_eff)

    valid_mask : np.ndarray | None
        Máscara booleana con las posiciones válidas de h_pilots.
        Shape: (M_eff, P_eff)

        Si no se proporciona, se considerarán válidas las posiciones no-NaN.

    use_conjugate_order : {"current_next", "next_current"}
        Define qué producto conjugado usar:

        - "current_next":
            S[m,p] = H[m,p] * conj(H[m+1,p])

        - "next_current":
            S[m,p] = H[m+1,p] * conj(H[m,p])

        Ambos son equivalentes salvo por el signo de la fase.

    return_mask : bool
        Si True, devuelve también la máscara de validez de S.

    Returns
    -------
    s_pilots : np.ndarray
        Matriz diferencial compleja.
        Shape: (M_eff-1, P_eff)

    s_valid_mask : np.ndarray, opcional
        Máscara booleana de posiciones válidas.
        Shape: (M_eff-1, P_eff)
    """
    h_pilots = np.asarray(h_pilots)

    if h_pilots.ndim != 2:
        raise ValueError(
            f"h_pilots debe tener shape (M_eff, P_eff), recibido {h_pilots.shape}"
        )

    if not np.iscomplexobj(h_pilots):
        raise TypeError("h_pilots debe ser complejo.")

    M_eff, P_eff = h_pilots.shape

    if M_eff < 2:
        raise ValueError("Se necesitan al menos dos símbolos para calcular diferencia de fase.")

    if valid_mask is None:
        valid_mask = ~np.isnan(h_pilots)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != h_pilots.shape:
            raise ValueError(
                f"valid_mask debe tener shape {h_pilots.shape}, recibido {valid_mask.shape}"
            )

    h0 = h_pilots[:-1, :]   # símbolo m
    h1 = h_pilots[1:, :]    # símbolo m+1

    m0 = valid_mask[:-1, :]
    m1 = valid_mask[1:, :]
    s_valid_mask = m0 & m1

    s_pilots = np.full((M_eff - 1, P_eff), np.nan + 1j * np.nan, dtype=np.complex128)

    if use_conjugate_order == "current_next":
        s_pilots[s_valid_mask] = h0[s_valid_mask] * np.conj(h1[s_valid_mask])

    elif use_conjugate_order == "next_current":
        s_pilots[s_valid_mask] = h1[s_valid_mask] * np.conj(h0[s_valid_mask])

    else:
        raise ValueError("use_conjugate_order debe ser 'current_next' o 'next_current'.")

    if return_mask:
        return s_pilots, s_valid_mask

    return s_pilots



def aggregate_pilot_slowtime(
    s_pilots: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    mode: str = "coherent_mean",
    min_valid_pilots: int = 1,
    return_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Agrega la matriz diferencial sobre pilotos para obtener una señal slow-time.

    Parámetros
    ----------
    s_pilots : np.ndarray
        Matriz compleja con la señal diferencial sobre pilotos.
        Shape esperada: (T, P)

        - T = número de instantes slow-time
        - P = número de pilotos

    valid_mask : np.ndarray | None
        Máscara booleana de posiciones válidas, shape (T, P).
        Si es None, se consideran válidas las posiciones no-NaN.

    mode : str
        Modo de agregación:
        - "coherent_mean"          -> media compleja
        - "noncoherent_mean_abs"   -> media de módulos
        - "noncoherent_mean_power" -> media de potencias
        - "keep_all"               -> no agrega, devuelve s_pilots

    min_valid_pilots : int
        Número mínimo de pilotos válidos exigidos por instante slow-time
        para considerar válida la salida agregada.

    return_mask : bool
        Si True, devuelve también una máscara de validez.

    Returns
    -------
    s_agg : np.ndarray
        Si mode != "keep_all":
            vector shape (T,)
        Si mode == "keep_all":
            matriz shape (T, P)

    agg_mask : np.ndarray, opcional
        Máscara booleana de validez:
        - shape (T,) si hay agregación
        - shape (T, P) si mode == "keep_all"
    """
    s_pilots = np.asarray(s_pilots)

    if s_pilots.ndim != 2:
        raise ValueError(f"s_pilots debe tener shape (T, P), recibido {s_pilots.shape}")

    T, P = s_pilots.shape

    if valid_mask is None:
        valid_mask = ~np.isnan(s_pilots)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != s_pilots.shape:
            raise ValueError(
                f"valid_mask debe tener shape {s_pilots.shape}, recibido {valid_mask.shape}"
            )

    if min_valid_pilots < 1 or min_valid_pilots > P:
        raise ValueError(f"min_valid_pilots debe estar entre 1 y {P}")

    if mode == "keep_all":
        if return_mask:
            return s_pilots, valid_mask
        return s_pilots

    counts = np.sum(valid_mask, axis=1)
    agg_mask = counts >= min_valid_pilots

    if mode == "coherent_mean":
        s_agg = np.full(T, np.nan + 1j * np.nan, dtype=np.complex128)
        for t in range(T):
            if agg_mask[t]:
                s_agg[t] = np.mean(s_pilots[t, valid_mask[t]])

    elif mode == "noncoherent_mean_abs":
        s_agg = np.full(T, np.nan, dtype=np.float64)
        for t in range(T):
            if agg_mask[t]:
                s_agg[t] = np.mean(np.abs(s_pilots[t, valid_mask[t]]))

    elif mode == "noncoherent_mean_power":
        s_agg = np.full(T, np.nan, dtype=np.float64)
        for t in range(T):
            if agg_mask[t]:
                vals = s_pilots[t, valid_mask[t]]
                s_agg[t] = np.mean(np.abs(vals) ** 2)

    else:
        raise ValueError(
            "mode debe ser 'coherent_mean', "
            "'noncoherent_mean_abs', 'noncoherent_mean_power' o 'keep_all'"
        )

    if return_mask:
        return s_agg, agg_mask

    return s_agg



def microdoppler_spectrogram(
    s_slow: np.ndarray,
    fs_slow: float,
    *,
    nperseg: int = 64,
    noverlap: int = 48,
    window: str = "hann",
    nfft: int | None = None,
    detrend: bool = False,
    center_frequency_axis: bool = True,
    to_db: bool = True,
    power_floor: float = 1e-12,
    return_complex_stft: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula el espectrograma micro-Doppler a partir de una señal slow-time.

    Parámetros
    ----------
    s_slow : np.ndarray
        Señal slow-time 1D.
        Puede ser compleja (caso coherente) o real (caso no coherente).

    fs_slow : float
        Frecuencia de muestreo en slow-time [Hz].
        Debe corresponder a la cadencia temporal efectiva de los pilotos o
        de las diferencias entre símbolos.

    nperseg : int
        Longitud de ventana STFT.

    noverlap : int
        Número de muestras solapadas entre ventanas consecutivas.

    window : str
        Tipo de ventana:
        - "hann"
        - "hamming"
        - "rect"

    nfft : int | None
        Tamaño de FFT por segmento.
        Si None, se toma nperseg.

    detrend : bool
        Si True, elimina la media de cada segmento antes de FFT.

    center_frequency_axis : bool
        Si True, aplica fftshift al eje de frecuencias y al espectrograma.

    to_db : bool
        Si True, convierte magnitud/potencia a dB.

    power_floor : float
        Suelo numérico para evitar log(0).

    return_complex_stft : bool
        Si True, devuelve además la STFT compleja.

    Returns
    -------
    spec : np.ndarray
        Espectrograma, shape (F, T_frames)

    f_axis : np.ndarray
        Eje de frecuencias Doppler [Hz], shape (F,)

    t_axis : np.ndarray
        Eje temporal del espectrograma [s], shape (T_frames,)

    Z : np.ndarray, opcional
        STFT compleja antes de magnitud/dB, shape (F, T_frames)
    """
    s_slow = np.asarray(s_slow)

    if s_slow.ndim != 1:
        raise ValueError(f"s_slow debe ser 1D, recibido shape {s_slow.shape}")

    if fs_slow <= 0:
        raise ValueError("fs_slow debe ser positivo.")

    if nperseg < 2:
        raise ValueError("nperseg debe ser al menos 2.")

    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap debe cumplir 0 <= noverlap < nperseg.")

    if nfft is None:
        nfft = nperseg

    if nfft < nperseg:
        raise ValueError("nfft debe ser >= nperseg.")

    if len(s_slow) < nperseg:
        raise ValueError(
            f"La señal slow-time tiene longitud {len(s_slow)} y es menor que nperseg={nperseg}."
        )

    # Definición de ventana
    if window == "hann":
        w = np.hanning(nperseg)
    elif window == "hamming":
        w = np.hamming(nperseg)
    elif window == "rect":
        w = np.ones(nperseg)
    else:
        raise ValueError("window debe ser 'hann', 'hamming' o 'rect'.")

    hop = nperseg - noverlap
    n_frames = 1 + (len(s_slow) - nperseg) // hop

    Z = np.zeros((nfft, n_frames), dtype=np.complex128)
    t_axis = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        start = i * hop
        stop = start + nperseg
        segment = s_slow[start:stop].astype(np.complex128, copy=False)

        if detrend:
            segment = segment - np.mean(segment)

        segment_w = segment * w
        Z[:, i] = np.fft.fft(segment_w, n=nfft)

        # tiempo asociado al centro de la ventana
        t_axis[i] = (start + nperseg / 2) / fs_slow

    # Eje de frecuencias
    f_axis = np.fft.fftfreq(nfft, d=1.0 / fs_slow)

    # Magnitud/potencia
    # Uso potencia espectral básica |Z|^2
    spec = np.abs(Z) ** 2

    if center_frequency_axis:
        spec = np.fft.fftshift(spec, axes=0)
        Z = np.fft.fftshift(Z, axes=0)
        f_axis = np.fft.fftshift(f_axis)

    if to_db:
        spec = 10.0 * np.log10(np.maximum(spec, power_floor))

    if return_complex_stft:
        return spec, f_axis, t_axis, Z

    return spec, f_axis, t_axis

@dataclass
class PhyPilotGridConfig:
    """
    Configuración de la rejilla OFDM propuesta por PHY.
    """
    fft_size: int = 128
    cp_len: int = 32
    symbols_per_slot: int = 14
    sync_symbol_index: int = 0
    sensing_pilot_symbol_index: int = 7
    active_subcarriers: np.ndarray = field(default_factory=lambda: np.arange(6, 121, dtype=int))
    scattered_pilot_period_time: int = 4
    scattered_pilot_period_freq: int = 8
    pilot_sequence_length: int = 113
    pilot_sequence_root: int = 25

    @property
    def symbol_time_s(self) -> float:
        # Fs = N * Δf = 128 * 30kHz = 3.84Msps -> Tsym = (128+32)/3.84e6
        return (self.fft_size + self.cp_len) / (self.fft_size * 30_000.0)

    @property
    def fs_sp(self) -> float:
        return 1.0 / (self.symbols_per_slot * self.symbol_time_s)


@dataclass
class ChannelEstimate:
    """
    Contenedor para H estimado en la rejilla OFDM.
    """
    h_grid: np.ndarray
    valid_mask: np.ndarray
    rx_grid: np.ndarray
    pilot_mask: np.ndarray
    pilot_values_grid: np.ndarray
    pilot_kind_grid: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MicroDopplerResult:
    """
    Resultado de alto nivel del pipeline micro-Doppler.
    """
    mode: str
    h_used: np.ndarray
    h_valid_mask: np.ndarray
    slow_matrix: np.ndarray
    slow_valid_mask: np.ndarray
    slow_agg: np.ndarray | None
    slow_agg_mask: np.ndarray | None
    spectrogram: np.ndarray
    f_axis: np.ndarray
    t_axis: np.ndarray
    fs_slow: float
    extra: dict[str, Any] = field(default_factory=dict)


def zadoff_chu_sequence(length: int, root: int) -> np.ndarray:
    """
    Genera una secuencia Zadoff-Chu compleja de longitud dada.
    """
    if length <= 0:
        raise ValueError("length debe ser positivo.")
    n = np.arange(length, dtype=np.float64)
    if length % 2 == 0:
        seq = np.exp(-1j * np.pi * root * n * n / length)
    else:
        seq = np.exp(-1j * np.pi * root * n * (n + 1.0) / length)
    return seq.astype(np.complex128)


def build_phy_pilot_grid(
    n_symbols_total: int,
    cfg: PhyPilotGridConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye la rejilla conocida de pilotos de la especificación PHY.

    Returns
    -------
    pilot_mask : np.ndarray of bool, shape (M, N)
    pilot_values_grid : np.ndarray complex, shape (M, N)
    pilot_kind_grid : np.ndarray str, shape (M, N)
        Valores: "none", "sync", "sp", "scattered"
    """
    if cfg is None:
        cfg = PhyPilotGridConfig()

    M = int(n_symbols_total)
    N = int(cfg.fft_size)
    active = np.asarray(cfg.active_subcarriers, dtype=int)

    pilot_mask = np.zeros((M, N), dtype=bool)
    pilot_values_grid = np.zeros((M, N), dtype=np.complex128)
    pilot_kind_grid = np.full((M, N), "none", dtype=object)

    zc = zadoff_chu_sequence(cfg.pilot_sequence_length, cfg.pilot_sequence_root)
    # Nota práctica: la especificación adjunta fija 115 activas y N_ZC=113.
    # Para no bloquear el pipeline, extendemos la ZC de forma periódica si hace falta.
    pilot_values_active = np.resize(zc, active.size).astype(np.complex128, copy=False)
    active_to_pos = {int(k): i for i, k in enumerate(active)}

    for m in range(M):
        m_local = m % cfg.symbols_per_slot

        if m_local == cfg.sync_symbol_index:
            pilot_mask[m, active] = True
            pilot_values_grid[m, active] = pilot_values_active
            pilot_kind_grid[m, active] = "sync"
            continue

        if m_local == cfg.sensing_pilot_symbol_index:
            pilot_mask[m, active] = True
            pilot_values_grid[m, active] = pilot_values_active
            pilot_kind_grid[m, active] = "sp"
            continue

        if m_local % cfg.scattered_pilot_period_time == 0:
            k_offset = ((m_local // cfg.scattered_pilot_period_time) % 2) * 4
            ks = active[(active % cfg.scattered_pilot_period_freq) == k_offset]
            pilot_mask[m, ks] = True
            pilot_kind_grid[m, ks] = "scattered"
            for k in ks:
                pilot_values_grid[m, k] = pilot_values_active[active_to_pos[int(k)]]

    return pilot_mask, pilot_values_grid, pilot_kind_grid


def estimate_sparse_channel_from_pilot_grid(
    rx_grid: np.ndarray,
    pilot_values_grid: np.ndarray,
    pilot_mask: np.ndarray,
    pilot_kind_grid: np.ndarray | None = None,
) -> ChannelEstimate:
    """
    Estima el canal únicamente en las celdas conocidas de piloto de una rejilla completa.
    """
    rx_grid = np.asarray(rx_grid)
    pilot_values_grid = np.asarray(pilot_values_grid)
    pilot_mask = np.asarray(pilot_mask, dtype=bool)

    if rx_grid.ndim != 2:
        raise ValueError("rx_grid debe tener shape (M, N).")
    if rx_grid.shape != pilot_values_grid.shape or rx_grid.shape != pilot_mask.shape:
        raise ValueError("rx_grid, pilot_values_grid y pilot_mask deben tener la misma shape.")

    h_grid = np.full(rx_grid.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    valid_mask = pilot_mask & (np.abs(pilot_values_grid) > 1e-12)
    h_grid[valid_mask] = rx_grid[valid_mask] / pilot_values_grid[valid_mask]

    if pilot_kind_grid is None:
        pilot_kind_grid = np.full(rx_grid.shape, "none", dtype=object)

    return ChannelEstimate(
        h_grid=h_grid,
        valid_mask=valid_mask,
        rx_grid=rx_grid,
        pilot_mask=pilot_mask,
        pilot_values_grid=pilot_values_grid,
        pilot_kind_grid=pilot_kind_grid,
        meta={
            "shape": tuple(rx_grid.shape),
            "num_valid": int(np.sum(valid_mask)),
        },
    )


def extract_sp_channel(
    channel_estimate: ChannelEstimate,
    cfg: PhyPilotGridConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrae H_sp[n_slot, k] usando únicamente los símbolos Sensing Pilot.

    Returns
    -------
    h_sp : np.ndarray, shape (N_slots, K_active)
    valid_mask : np.ndarray, shape (N_slots, K_active)
    sp_symbol_indices : np.ndarray, shape (N_slots,)
    """
    if cfg is None:
        cfg = PhyPilotGridConfig()

    active = np.asarray(cfg.active_subcarriers, dtype=int)
    M, _ = channel_estimate.h_grid.shape
    sp_symbol_indices = np.arange(cfg.sensing_pilot_symbol_index, M, cfg.symbols_per_slot, dtype=int)

    h_sp = channel_estimate.h_grid[np.ix_(sp_symbol_indices, active)]
    valid_mask = channel_estimate.valid_mask[np.ix_(sp_symbol_indices, active)]
    return h_sp, valid_mask, sp_symbol_indices


def _interp_1d_complex(x_known: np.ndarray, y_known: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    y_real = np.interp(x_target, x_known, np.real(y_known))
    y_imag = np.interp(x_target, x_known, np.imag(y_known))
    return y_real + 1j * y_imag


def interpolate_channel_2d_linear(
    h_sparse: np.ndarray,
    valid_mask: np.ndarray,
    *,
    active_subcarriers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolación 2D lineal conservadora:
    - primero en frecuencia dentro de cada símbolo
    - después en tiempo para cada subportadora

    Solo se rellenan celdas dentro del rectángulo activo definido por las muestras disponibles.
    """
    h_sparse = np.asarray(h_sparse)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if h_sparse.shape != valid_mask.shape or h_sparse.ndim != 2:
        raise ValueError("h_sparse y valid_mask deben tener la misma shape 2D.")

    M, N = h_sparse.shape
    if active_subcarriers is None:
        active_subcarriers = np.arange(N, dtype=int)
    active_subcarriers = np.asarray(active_subcarriers, dtype=int)

    h_freq = np.full((M, N), np.nan + 1j * np.nan, dtype=np.complex128)
    mask_freq = np.zeros((M, N), dtype=bool)

    # 1) Interpolación en frecuencia, símbolo a símbolo
    for m in range(M):
        ks = active_subcarriers[valid_mask[m, active_subcarriers]]
        if ks.size == 0:
            continue
        if ks.size == 1:
            h_freq[m, ks[0]] = h_sparse[m, ks[0]]
            mask_freq[m, ks[0]] = True
            continue

        kmin = int(np.min(ks))
        kmax = int(np.max(ks))
        k_target = active_subcarriers[(active_subcarriers >= kmin) & (active_subcarriers <= kmax)]
        h_freq[m, k_target] = _interp_1d_complex(ks, h_sparse[m, ks], k_target)
        mask_freq[m, k_target] = True

    # Mantener exactamente las observaciones originales
    h_freq[valid_mask] = h_sparse[valid_mask]
    mask_freq[valid_mask] = True

    # 2) Interpolación temporal, subportadora a subportadora
    h_full = np.full((M, N), np.nan + 1j * np.nan, dtype=np.complex128)
    full_mask = np.zeros((M, N), dtype=bool)

    t_all = np.arange(M, dtype=int)
    for k in active_subcarriers:
        ts = t_all[mask_freq[:, k]]
        if ts.size == 0:
            continue
        if ts.size == 1:
            h_full[ts[0], k] = h_freq[ts[0], k]
            full_mask[ts[0], k] = True
            continue

        tmin = int(np.min(ts))
        tmax = int(np.max(ts))
        t_target = t_all[(t_all >= tmin) & (t_all <= tmax)]
        h_full[t_target, k] = _interp_1d_complex(ts, h_freq[ts, k], t_target)
        full_mask[t_target, k] = True

    h_full[valid_mask] = h_sparse[valid_mask]
    full_mask[valid_mask] = True
    return h_full, full_mask


def clutter_suppress_slowtime(
    x: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    mode: str = "mean_subtract",
    ema_alpha: float = 0.95,
    return_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Filtrado simple de clutter sobre el eje slow-time (axis=0).
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x debe tener shape (T, P).")

    if valid_mask is None:
        valid_mask = ~np.isnan(x)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != x.shape:
            raise ValueError("valid_mask debe tener la misma shape que x.")

    y = np.full_like(x, np.nan + 1j * np.nan, dtype=np.complex128)

    if mode == "none":
        y[valid_mask] = x[valid_mask]

    elif mode == "mean_subtract":
        for p in range(x.shape[1]):
            mask_p = valid_mask[:, p]
            if np.any(mask_p):
                mu = np.mean(x[mask_p, p])
                y[mask_p, p] = x[mask_p, p] - mu

    elif mode == "first_difference":
        diff = np.full_like(x, np.nan + 1j * np.nan, dtype=np.complex128)
        diff_mask = np.zeros_like(valid_mask, dtype=bool)
        for p in range(x.shape[1]):
            for t in range(1, x.shape[0]):
                if valid_mask[t, p] and valid_mask[t - 1, p]:
                    diff[t, p] = x[t, p] - x[t - 1, p]
                    diff_mask[t, p] = True
        y = diff
        valid_mask = diff_mask

    elif mode == "ema_highpass":
        for p in range(x.shape[1]):
            ema = None
            for t in range(x.shape[0]):
                if not valid_mask[t, p]:
                    continue
                val = x[t, p]
                if ema is None:
                    ema = val
                else:
                    ema = ema_alpha * ema + (1.0 - ema_alpha) * val
                y[t, p] = val - ema

    else:
        raise ValueError("mode debe ser 'none', 'mean_subtract', 'first_difference' o 'ema_highpass'.")

    if return_mask:
        return y, valid_mask
    return y


def _extract_rx_grid_from_source(
    source_type: Literal["sionna", "usrp_iq"],
    *,
    data: Any | None = None,
    file_path: str | Path | None = None,
    data_kind: Literal["grid", "time"] | None = None,
    fft_size: int | None = None,
    cp_len: int | None = None,
    n_symbols: int | None = None,
    usrp_file_dtype: Literal["complex64", "sc16_interleaved"] = "complex64",
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Obtiene rx_grid completa a partir de entrada Sionna o IQ USRP.
    """
    if source_type == "sionna":
        if data is None:
            raise ValueError("Para source_type='sionna' debes pasar `data`.")
        if data_kind is None:
            raise ValueError("Para Sionna debes indicar data_kind='grid' o 'time'.")

        arr = np.asarray(data)
        if data_kind == "grid":
            return _to_grid_mn(_ensure_complex_ndarray(arr, "data")), None
        if data_kind == "time":
            if fft_size is None or cp_len is None:
                raise ValueError("Para data_kind='time' debes pasar fft_size y cp_len.")
            samples_td = _ensure_complex_ndarray(arr, "data").reshape(-1)
            return _remove_cp_and_fft(samples_td, fft_size, cp_len, n_symbols=n_symbols), samples_td
        raise ValueError(f"data_kind no soportado para Sionna: {data_kind}")

    if source_type == "usrp_iq":
        if fft_size is None or cp_len is None:
            raise ValueError("Para source_type='usrp_iq' debes pasar fft_size y cp_len.")
        if file_path is None and data is None:
            raise ValueError("Debes pasar `file_path` o `data` para USRP IQ.")
        if file_path is not None:
            samples_td = _read_usrp_iq_file(file_path, dtype=usrp_file_dtype)
        else:
            samples_td = _ensure_complex_ndarray(data, "data").reshape(-1)
        return _remove_cp_and_fft(samples_td, fft_size, cp_len, n_symbols=n_symbols), samples_td

    raise ValueError(f"source_type no soportado: {source_type}")


def run_microdoppler_pipeline(
    source_type: Literal["sionna", "usrp_iq"],
    *,
    mode: Literal["sp_only", "full_interpolated"] = "sp_only",
    data: Any | None = None,
    file_path: str | Path | None = None,
    data_kind: Literal["grid", "time"] | None = None,
    n_symbols: int | None = None,
    usrp_file_dtype: Literal["complex64", "sc16_interleaved"] = "complex64",
    cfg: PhyPilotGridConfig | None = None,
    differential: bool = True,
    differential_order: Literal["current_next", "next_current"] = "current_next",
    clutter_mode: Literal["none", "mean_subtract", "first_difference", "ema_highpass"] = "mean_subtract",
    ema_alpha: float = 0.95,
    aggregate_mode: Literal["coherent_mean", "noncoherent_mean_abs", "noncoherent_mean_power", "keep_all", "none"] = "coherent_mean",
    min_valid_pilots: int = 1,
    nperseg: int = 64,
    noverlap: int = 48,
    window: str = "hann",
    nfft: int | None = None,
    detrend: bool = False,
    to_db: bool = True,
    power_floor: float = 1e-12,
) -> MicroDopplerResult:
    """
    Pipeline de alto nivel compatible con dos modos:

    - sp_only:
        usa exclusivamente H_sp[n_slot, k] como serie principal de Doppler.

    - full_interpolated:
        estima H en todos los pilotos conocidos (SYNC + SP + dispersos),
        interpola 2D y luego sigue el pipeline diferencial ya establecido.
    """
    if cfg is None:
        cfg = PhyPilotGridConfig()

    rx_grid, samples_td = _extract_rx_grid_from_source(
        source_type,
        data=data,
        file_path=file_path,
        data_kind=data_kind,
        fft_size=cfg.fft_size,
        cp_len=cfg.cp_len,
        n_symbols=n_symbols,
        usrp_file_dtype=usrp_file_dtype,
    )

    pilot_mask, pilot_values_grid, pilot_kind_grid = build_phy_pilot_grid(rx_grid.shape[0], cfg)
    ch = estimate_sparse_channel_from_pilot_grid(rx_grid, pilot_values_grid, pilot_mask, pilot_kind_grid)

    if mode == "sp_only":
        h_used, h_valid_mask, sp_symbol_indices = extract_sp_channel(ch, cfg)
        fs_slow = cfg.fs_sp
    elif mode == "full_interpolated":
        h_used, h_valid_mask = interpolate_channel_2d_linear(
            ch.h_grid,
            ch.valid_mask,
            active_subcarriers=cfg.active_subcarriers,
        )
        sp_symbol_indices = np.arange(cfg.sensing_pilot_symbol_index, rx_grid.shape[0], cfg.symbols_per_slot, dtype=int)
        fs_slow = 1.0 / cfg.symbol_time_s
    else:
        raise ValueError("mode debe ser 'sp_only' o 'full_interpolated'.")

    # Restricción a subportadoras activas
    active = np.asarray(cfg.active_subcarriers, dtype=int)
    h_used = h_used[:, active] if h_used.shape[1] == cfg.fft_size else h_used
    h_valid_mask = h_valid_mask[:, active] if h_valid_mask.shape[1] == cfg.fft_size else h_valid_mask

    # Clutter suppression opcional sobre H antes del diferencial
    h_filt, h_filt_mask = clutter_suppress_slowtime(
        h_used,
        h_valid_mask,
        mode=clutter_mode,
        ema_alpha=ema_alpha,
        return_mask=True,
    )

    if differential:
        slow_matrix, slow_valid_mask = phase_difference_on_pilots(
            h_filt,
            h_filt_mask,
            use_conjugate_order=differential_order,
            return_mask=True,
        )
        if mode == "sp_only":
            fs_slow = cfg.fs_sp
        else:
            fs_slow = 1.0 / cfg.symbol_time_s
    else:
        slow_matrix, slow_valid_mask = h_filt, h_filt_mask

    slow_agg = None
    slow_agg_mask = None

    if aggregate_mode == "none":
        if slow_matrix.shape[1] == 0:
            raise ValueError("No hay subportadoras para STFT.")
        first_valid = np.argmax(np.sum(slow_valid_mask, axis=0))
        slow_agg = np.asarray(slow_matrix[:, first_valid])
        slow_agg_mask = np.asarray(slow_valid_mask[:, first_valid])
    else:
        agg_mode_internal = "keep_all" if aggregate_mode == "keep_all" else aggregate_mode
        agg_out, agg_mask = aggregate_pilot_slowtime(
            slow_matrix,
            slow_valid_mask,
            mode=agg_mode_internal,
            min_valid_pilots=min_valid_pilots,
            return_mask=True,
        )
        if aggregate_mode == "keep_all":
            # escoger subportadora activa con mayor número de válidas para la STFT visible
            counts = np.sum(agg_mask, axis=0)
            k_best = int(np.argmax(counts))
            slow_agg = agg_out[:, k_best]
            slow_agg_mask = agg_mask[:, k_best]
        else:
            slow_agg = agg_out
            slow_agg_mask = agg_mask

    if slow_agg_mask is None:
        raise RuntimeError("No se ha construido slow_agg_mask.")

    slow_for_stft = np.asarray(slow_agg[slow_agg_mask])
    if slow_for_stft.ndim != 1:
        slow_for_stft = slow_for_stft.reshape(-1)

    spec, f_axis, t_axis = microdoppler_spectrogram(
        slow_for_stft,
        fs_slow=fs_slow,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        nfft=nfft,
        detrend=detrend,
        to_db=to_db,
        power_floor=power_floor,
    )

    return MicroDopplerResult(
        mode=mode,
        h_used=h_used,
        h_valid_mask=h_valid_mask,
        slow_matrix=slow_matrix,
        slow_valid_mask=slow_valid_mask,
        slow_agg=slow_agg,
        slow_agg_mask=slow_agg_mask,
        spectrogram=spec,
        f_axis=f_axis,
        t_axis=t_axis,
        fs_slow=fs_slow,
        extra={
            "samples_td": samples_td,
            "pilot_mask": pilot_mask,
            "pilot_values_grid": pilot_values_grid,
            "pilot_kind_grid": pilot_kind_grid,
            "sp_symbol_indices": sp_symbol_indices,
            "symbol_time_s": cfg.symbol_time_s,
            "num_valid_sparse": int(np.sum(ch.valid_mask)),
        },
    )
