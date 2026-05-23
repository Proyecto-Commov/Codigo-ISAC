from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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

IQConcreteFormat = Literal[
    "complex64",
    "sc16_interleaved",
    "sc8_interleaved",
]

DistanceMode = Literal["monostatic", "bistatic"]
WindowName = Literal["rect", "hann", "hamming"]


# =============================================================================
# Configuración YAML
# =============================================================================


def load_modulator_yaml(yaml_path: str | Path) -> dict[str, Any]:
    """
    Lee Modulator.yaml y devuelve su contenido como diccionario.
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"No existe el archivo YAML: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("El YAML no contiene un diccionario válido.")

    return cfg


@dataclass(frozen=True)
class OFDMParams:
    """
    Parámetros OFDM básicos leídos desde Modulator.yaml.
    """
    fft_size: int
    cp_len: int
    num_symbols: int
    sample_rate: float
    sensing_symbol_stride: int
    range_fft_size: int
    doppler_fft_size: int
    pilot_positions: np.ndarray
    center_freq: float | None = None

    @property
    def samples_per_symbol(self) -> int:
        return self.fft_size + self.cp_len

    @property
    def samples_per_frame(self) -> int:
        return self.samples_per_symbol * self.num_symbols

    @property
    def subcarrier_spacing(self) -> float:
        return self.sample_rate / self.fft_size

    @property
    def fs_slow(self) -> float:
        return self.sample_rate / (self.sensing_symbol_stride * self.samples_per_symbol)


def get_ofdm_params(cfg: dict[str, Any]) -> OFDMParams:
    """
    Extrae y normaliza los parámetros OFDM usados por el pipeline.
    """
    fft_size = int(cfg["fft_size"])
    cp_len = int(cfg["cp_length"])
    num_symbols = int(cfg.get("num_symbols", cfg.get("sensing_symbol_num", 1)))
    sample_rate = float(cfg["sample_rate"])
    sensing_symbol_stride = int(cfg.get("sensing_symbol_stride", 1))
    range_fft_size = int(cfg.get("range_fft_size", fft_size))
    doppler_fft_size = int(cfg.get("doppler_fft_size", num_symbols))
    pilot_positions = np.asarray(cfg.get("pilot_positions", []), dtype=int)
    center_freq = float(cfg["center_freq"]) if "center_freq" in cfg else None

    if fft_size <= 0:
        raise ValueError("fft_size debe ser positivo.")
    if cp_len < 0:
        raise ValueError("cp_length no puede ser negativo.")
    if num_symbols <= 0:
        raise ValueError("num_symbols debe ser positivo.")
    if sample_rate <= 0:
        raise ValueError("sample_rate debe ser positivo.")
    if sensing_symbol_stride <= 0:
        raise ValueError("sensing_symbol_stride debe ser positivo.")
    if range_fft_size < fft_size:
        raise ValueError("range_fft_size debe ser >= fft_size.")
    if doppler_fft_size <= 0:
        raise ValueError("doppler_fft_size debe ser positivo.")
    if pilot_positions.size > 0:
        if np.any((pilot_positions < 0) | (pilot_positions >= fft_size)):
            raise ValueError("pilot_positions contiene índices fuera de rango.")

    return OFDMParams(
        fft_size=fft_size,
        cp_len=cp_len,
        num_symbols=num_symbols,
        sample_rate=sample_rate,
        sensing_symbol_stride=sensing_symbol_stride,
        range_fft_size=range_fft_size,
        doppler_fft_size=doppler_fft_size,
        pilot_positions=pilot_positions,
        center_freq=center_freq,
    )


def compute_fs_slow_from_cfg(cfg: dict[str, Any]) -> float:
    """
    Frecuencia de muestreo slow-time [Hz] a partir del YAML.
    """
    return get_ofdm_params(cfg).fs_slow


def compute_range_axis_from_cfg(
    cfg: dict[str, Any],
    n_range_bins: int | None = None,
    *,
    distance_mode: DistanceMode = "monostatic",
    c: float = 299_792_458.0,
) -> np.ndarray:
    """
    Calcula eje de distancia [m] asociado a los bins de retardo.

    En monostático: R = c*tau/2.
    En bistático: distancia excedente = c*tau.
    """
    params = get_ofdm_params(cfg)
    if n_range_bins is None:
        n_range_bins = params.range_fft_size
    n_range_bins = int(n_range_bins)

    if n_range_bins <= 0:
        raise ValueError("n_range_bins debe ser positivo.")

    tau_bins = np.arange(n_range_bins, dtype=np.float64)
    tau_axis = tau_bins / (n_range_bins * params.subcarrier_spacing)

    if distance_mode == "monostatic":
        return c * tau_axis / 2.0
    if distance_mode == "bistatic":
        return c * tau_axis
    raise ValueError("distance_mode debe ser 'monostatic' o 'bistatic'.")


# =============================================================================
# Lectura IQ y demodulación OFDM
# =============================================================================


def _ensure_complex_ndarray(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if not np.iscomplexobj(arr):
        raise TypeError(f"{name} debe ser complejo.")
    return arr


def _read_iq_with_format(bin_path: str | Path, fmt: IQConcreteFormat) -> np.ndarray:
    """
    Lee un .bin IQ y devuelve iq[n] = I[n] + jQ[n].
    """
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"No existe el archivo binario: {bin_path}")

    if fmt == "complex64":
        return np.fromfile(bin_path, dtype=np.complex64)

    if fmt == "sc16_interleaved":
        raw = np.fromfile(bin_path, dtype=np.int16)
        if raw.size % 2 != 0:
            raise ValueError("El archivo sc16_interleaved tiene número impar de enteros.")
        i = raw[0::2].astype(np.float32)
        q = raw[1::2].astype(np.float32)
        return ((i + 1j * q) / 32768.0).astype(np.complex64)

    if fmt == "sc8_interleaved":
        raw = np.fromfile(bin_path, dtype=np.int8)
        if raw.size % 2 != 0:
            raise ValueError("El archivo sc8_interleaved tiene número impar de enteros.")
        i = raw[0::2].astype(np.float32)
        q = raw[1::2].astype(np.float32)
        return ((i + 1j * q) / 128.0).astype(np.complex64)

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
    Métrica simple de coherencia CP-cola para inferir el formato IQ.
    """
    iq = np.asarray(iq)
    if iq.ndim != 1:
        raise ValueError("iq debe ser 1D.")
    if cp_len <= 0:
        return 0.0

    sym_len = fft_size + cp_len
    n_symbols = min(len(iq) // sym_len, max_symbols)
    if n_symbols <= 0:
        return 0.0

    x = iq[: n_symbols * sym_len].reshape(n_symbols, sym_len)
    cp = x[:, :cp_len]
    tail = x[:, -cp_len:]

    num = np.sum(cp * np.conj(tail), axis=1)
    den = np.sqrt(np.sum(np.abs(cp) ** 2, axis=1) * np.sum(np.abs(tail) ** 2, axis=1)) + eps
    return float(np.nanmedian(np.abs(num / den)))


def infer_iq_storage_format(bin_path: str | Path, yaml_path: str | Path) -> IQConcreteFormat:
    """
    Intenta inferir el formato de almacenamiento del .bin mediante coherencia CP-cola.
    """
    params = get_ofdm_params(load_modulator_yaml(yaml_path))
    candidates: list[IQConcreteFormat] = ["complex64", "sc16_interleaved", "sc8_interleaved"]
    scores: dict[str, float] = {}

    for fmt in candidates:
        try:
            iq = _read_iq_with_format(bin_path, fmt)
            scores[fmt] = _cp_tail_correlation_score(
                iq,
                fft_size=params.fft_size,
                cp_len=params.cp_len,
            )
        except Exception:
            scores[fmt] = -np.inf

    best_fmt = max(scores, key=scores.get)
    if not np.isfinite(scores[best_fmt]):
        raise ValueError("No se ha podido inferir un formato IQ válido.")
    return best_fmt  # type: ignore[return-value]


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
    Lee IQ crudo de USRP y devuelve la señal compleja temporal.
    """
    cfg = load_modulator_yaml(yaml_path)
    params = get_ofdm_params(cfg)

    fmt = infer_iq_storage_format(bin_path, yaml_path) if storage_format == "auto" else storage_format
    if fmt == "auto":
        raise ValueError("Formato IQ inválido tras inferencia.")

    iq = _read_iq_with_format(bin_path, fmt)  # type: ignore[arg-type]

    if trim_to_complete_frames:
        n_complete_frames = len(iq) // params.samples_per_frame
        if n_complete_frames == 0:
            raise ValueError("El archivo no contiene ni una trama completa según el YAML.")
        iq = iq[: n_complete_frames * params.samples_per_frame]

    t = np.arange(len(iq), dtype=np.float64) / params.sample_rate

    if verbose:
        print("Archivo IQ leído correctamente")
        print(f"  Ruta: {bin_path}")
        print(f"  Formato detectado/usado: {fmt}")
        print(f"  Muestras complejas: {len(iq)}")
        print(f"  Sample rate: {params.sample_rate:.3f} Hz")
        print(f"  Duración: {len(iq) / params.sample_rate:.9f} s")
        print(f"  Muestras por símbolo OFDM: {params.samples_per_symbol}")
        print(f"  Muestras por trama OFDM: {params.samples_per_frame}")
        print(f"  Símbolos completos: {len(iq) // params.samples_per_symbol}")
        print(f"  Tramas completas: {len(iq) // params.samples_per_frame}")
        print(f"  dtype salida: {iq.dtype}")

    return (iq, t) if return_time_axis else iq


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
    Demodula OFDM temporal mediante CP removal + FFT.

    Salida: grid.shape = (M, N), con grid[m,k].
    """
    params = get_ofdm_params(load_modulator_yaml(yaml_path))
    iq = _ensure_complex_ndarray(iq, "iq").reshape(-1)

    if start_sample < 0 or start_sample >= len(iq):
        raise ValueError("start_sample está fuera del vector iq.")

    iq_work = iq[start_sample:]
    total_complete_symbols = len(iq_work) // params.samples_per_symbol
    if total_complete_symbols <= 0:
        raise ValueError("No hay suficientes muestras para extraer un símbolo OFDM completo.")

    n_symbols_used = total_complete_symbols if n_symbols is None else min(int(n_symbols), total_complete_symbols)
    if n_symbols_used <= 0:
        raise ValueError("n_symbols debe ser positivo.")

    iq_used = iq_work[: n_symbols_used * params.samples_per_symbol]
    symbols_with_cp = iq_used.reshape(n_symbols_used, params.samples_per_symbol)
    symbols_no_cp = symbols_with_cp[:, params.cp_len:]

    grid = np.fft.fft(symbols_no_cp, n=params.fft_size, axis=1)
    if normalize_fft:
        grid = grid / np.sqrt(params.fft_size)
    if fftshift:
        grid = np.fft.fftshift(grid, axes=1)

    symbol_time_axis = (
        start_sample / params.sample_rate
        + np.arange(n_symbols_used, dtype=np.float64) * params.samples_per_symbol / params.sample_rate
    )
    subcarrier_axis = np.fft.fftfreq(params.fft_size, d=1.0 / params.sample_rate)
    if fftshift:
        subcarrier_axis = np.fft.fftshift(subcarrier_axis)

    if verbose:
        print("Demodulación OFDM completada")
        print(f"  FFT size: {params.fft_size}")
        print(f"  CP length: {params.cp_len}")
        print(f"  Símbolos demodulados: {n_symbols_used}")
        print(f"  Shape rejilla OFDM: {grid.shape}")
        print(f"  fftshift: {fftshift}")
        print(f"  normalize_fft: {normalize_fft}")

    return (grid, symbol_time_axis, subcarrier_axis) if return_axes else grid


# =============================================================================
# Adaptación a pilotos y estimación de canal
# =============================================================================


@dataclass
class PilotObservation:
    """
    Observaciones recibidas solo en pilotos.

    y_pilots[i,j] = Y[m_i, k_j], con shape (M_eff, P_eff).
    """
    source: Literal["sionna", "usrp_iq", "grid"]
    domain: Literal["freq", "mixed"]
    y_pilots: np.ndarray
    pilot_subcarriers: np.ndarray
    pilot_symbol_indices: np.ndarray
    rx_grid: np.ndarray | None = None
    samples_td: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _to_grid_mn(arr: np.ndarray) -> np.ndarray:
    """
    Convierte entradas típicas Sionna a shape (M,N).

    Soporta:
    - (M,N)
    - (batch,M,N)
    - (batch,tx,streams,M,N)
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 5:
        return arr[0, 0, 0]
    raise ValueError(f"No puedo interpretar shape {arr.shape} como rejilla OFDM.")


def _validate_pilot_spec(
    rx_grid: np.ndarray,
    pilot_subcarriers: np.ndarray,
    pilot_symbol_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    M, N = rx_grid.shape
    pilot_subcarriers = np.asarray(pilot_subcarriers, dtype=int).reshape(-1)
    if pilot_subcarriers.size == 0:
        raise ValueError("pilot_subcarriers no puede estar vacío.")
    if np.any((pilot_subcarriers < 0) | (pilot_subcarriers >= N)):
        raise IndexError(f"Hay subportadoras piloto fuera de rango [0,{N-1}].")

    if pilot_symbol_indices is None:
        pilot_symbol_indices = np.arange(M, dtype=int)
    else:
        pilot_symbol_indices = np.asarray(pilot_symbol_indices, dtype=int).reshape(-1)
        if pilot_symbol_indices.size == 0:
            raise ValueError("pilot_symbol_indices no puede estar vacío.")
        if np.any((pilot_symbol_indices < 0) | (pilot_symbol_indices >= M)):
            raise IndexError(f"Hay símbolos piloto fuera de rango [0,{M-1}].")

    return pilot_subcarriers, pilot_symbol_indices


def extract_pilot_observations_from_grid(
    rx_grid: np.ndarray,
    *,
    pilot_subcarriers: np.ndarray,
    pilot_symbol_indices: np.ndarray | None = None,
    keep_rx_grid: bool = False,
) -> PilotObservation:
    """
    Extrae Y[m,k] solo en posiciones piloto desde una rejilla OFDM ya demodulada.
    """
    rx_grid = _to_grid_mn(_ensure_complex_ndarray(rx_grid, "rx_grid"))
    pilot_subcarriers, pilot_symbol_indices = _validate_pilot_spec(
        rx_grid, pilot_subcarriers, pilot_symbol_indices
    )
    y_pilots = rx_grid[np.ix_(pilot_symbol_indices, pilot_subcarriers)]

    return PilotObservation(
        source="grid",
        domain="freq",
        y_pilots=y_pilots,
        pilot_subcarriers=pilot_subcarriers,
        pilot_symbol_indices=pilot_symbol_indices,
        rx_grid=rx_grid if keep_rx_grid else None,
        meta={"shape_rx_grid": tuple(rx_grid.shape), "shape_y_pilots": tuple(y_pilots.shape)},
    )


def adapt_to_pilot_observations(
    source_type: Literal["sionna", "usrp_iq"],
    *,
    yaml_path: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
    data: Any | None = None,
    file_path: str | Path | None = None,
    data_kind: Literal["grid", "time"] | None = None,
    pilot_subcarriers: np.ndarray | None = None,
    pilot_symbol_indices: np.ndarray | None = None,
    n_symbols: int | None = None,
    usrp_file_dtype: IQStorageFormat = "auto",
    keep_rx_grid: bool = False,
) -> PilotObservation:
    """
    Adaptador de entrada: Sionna o USRP IQ -> Y en pilotos.

    Si pilot_subcarriers=None, se usan cfg['pilot_positions'].
    """
    if cfg is None:
        if yaml_path is None:
            raise ValueError("Debes pasar cfg o yaml_path.")
        cfg = load_modulator_yaml(yaml_path)
    params = get_ofdm_params(cfg)

    if pilot_subcarriers is None:
        if params.pilot_positions.size == 0:
            raise ValueError("No hay pilot_positions en el YAML; pasa pilot_subcarriers.")
        pilot_subcarriers = params.pilot_positions

    if source_type == "sionna":
        if data is None:
            raise ValueError("Para source_type='sionna' debes pasar data.")
        if data_kind is None:
            raise ValueError("Para Sionna debes indicar data_kind='grid' o 'time'.")

        arr = np.asarray(data)
        samples_td = None
        if data_kind == "grid":
            rx_grid = _to_grid_mn(_ensure_complex_ndarray(arr, "data"))
            domain: Literal["freq", "mixed"] = "freq"
        elif data_kind == "time":
            samples_td = _ensure_complex_ndarray(arr, "data").reshape(-1)
            rx_grid = _remove_cp_and_fft(samples_td, params.fft_size, params.cp_len, n_symbols=n_symbols)
            domain = "mixed"
        else:
            raise ValueError(f"data_kind no soportado: {data_kind}")

        obs = extract_pilot_observations_from_grid(
            rx_grid,
            pilot_subcarriers=pilot_subcarriers,
            pilot_symbol_indices=pilot_symbol_indices,
            keep_rx_grid=keep_rx_grid,
        )
        obs.source = "sionna"
        obs.domain = domain
        obs.samples_td = samples_td
        obs.meta.update({"input_kind": f"sionna_{data_kind}"})
        return obs

    if source_type == "usrp_iq":
        if file_path is None and data is None:
            raise ValueError("Para USRP debes pasar file_path o data.")
        if file_path is not None:
            if yaml_path is None and usrp_file_dtype == "auto":
                raise ValueError("Para inferir formato IQ automáticamente necesitas yaml_path.")
            fmt = infer_iq_storage_format(file_path, yaml_path) if usrp_file_dtype == "auto" else usrp_file_dtype
            samples_td = _read_iq_with_format(file_path, fmt)  # type: ignore[arg-type]
        else:
            samples_td = _ensure_complex_ndarray(data, "data").reshape(-1)
            fmt = "array"

        rx_grid = _remove_cp_and_fft(samples_td, params.fft_size, params.cp_len, n_symbols=n_symbols)
        obs = extract_pilot_observations_from_grid(
            rx_grid,
            pilot_subcarriers=pilot_subcarriers,
            pilot_symbol_indices=pilot_symbol_indices,
            keep_rx_grid=keep_rx_grid,
        )
        obs.source = "usrp_iq"
        obs.domain = "mixed"
        obs.samples_td = samples_td
        obs.meta.update({"input_kind": "usrp_iq", "usrp_file_dtype": fmt})
        return obs

    raise ValueError(f"source_type no soportado: {source_type}")


def estimate_channel_on_pilots(
    y_pilots: np.ndarray,
    pilot_values: np.ndarray,
    *,
    eps: float = 1e-12,
    return_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Estima H en pilotos: H_pilots = Y_pilots / X_pilots.

    pilot_values puede tener shape (P,) o (M,P).
    """
    y_pilots = _ensure_complex_ndarray(y_pilots, "y_pilots")
    pilot_values = _ensure_complex_ndarray(pilot_values, "pilot_values")

    if y_pilots.ndim != 2:
        raise ValueError(f"y_pilots debe tener shape (M,P), recibido {y_pilots.shape}")

    M, P = y_pilots.shape
    if pilot_values.ndim == 1:
        if pilot_values.shape[0] != P:
            raise ValueError(f"pilot_values tiene longitud {pilot_values.shape[0]}, pero P={P}.")
        x_pilots = np.broadcast_to(pilot_values.reshape(1, P), (M, P))
    elif pilot_values.ndim == 2:
        if pilot_values.shape != (M, P):
            raise ValueError(f"pilot_values debe tener shape {(M, P)}, recibido {pilot_values.shape}.")
        x_pilots = pilot_values
    else:
        raise ValueError("pilot_values debe tener shape (P,) o (M,P).")

    valid_mask = np.abs(x_pilots) > eps
    H = np.full((M, P), np.nan + 1j * np.nan, dtype=np.complex128)
    H[valid_mask] = y_pilots[valid_mask] / x_pilots[valid_mask]

    return (H, valid_mask) if return_mask else H


def estimate_channel_grid(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    eps: float = 1e-12,
    return_mask: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Estima H en toda la rejilla: H = Y/X.
    """
    X = _ensure_complex_ndarray(X, "X")
    Y = _ensure_complex_ndarray(Y, "Y")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X e Y deben tener shape (M,N).")
    if X.shape != Y.shape:
        raise ValueError(f"X e Y deben tener la misma shape. X={X.shape}, Y={Y.shape}")

    valid_mask = np.abs(X) > eps
    H = np.full(X.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    H[valid_mask] = Y[valid_mask] / X[valid_mask]

    if verbose:
        print("Estimación de canal completada")
        print(f"  Shape H: {H.shape}")
        print(f"  Posiciones válidas: {np.sum(valid_mask)}/{valid_mask.size}")

    return (H, valid_mask) if return_mask else H


# =============================================================================
# Procesado OpenISAC: retardo-tiempo, rango-Doppler y micro-Doppler
# =============================================================================


def _replace_invalid_with_zero(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def delay_time_ifft_from_H(
    H: np.ndarray,
    *,
    modulator_cfg: dict[str, Any],
    n_subcarriers: int | None = None,
    pilot_subcarriers: np.ndarray | None = None,
    range_fft_size: int | None = None,
    apply_ifftshift: bool = False,
    distance_mode: DistanceMode = "monostatic",
    c: float = 299_792_458.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    IFFT sobre subportadoras: H[m,k] -> h[tau,m].

    Si H solo contiene pilotos, pasa pilot_subcarriers y se colocan en una rejilla
    completa rellenando el resto con ceros.

    Devuelve:
    - h_tau_m.shape = (N_range, M)
    - range_axis en metros.
    """
    H = _ensure_complex_ndarray(H, "H")
    if H.ndim != 2:
        raise ValueError(f"H debe tener shape (M,N) o (M,P), recibido {H.shape}")

    params = get_ofdm_params(modulator_cfg)
    M, K = H.shape

    if pilot_subcarriers is None:
        H_grid = H.astype(np.complex128, copy=False)
        N = K
    else:
        pilot_subcarriers = np.asarray(pilot_subcarriers, dtype=int).reshape(-1)
        if pilot_subcarriers.size != K:
            raise ValueError(
                f"H tiene {K} columnas, pero pilot_subcarriers tiene {pilot_subcarriers.size}."
            )
        N = params.fft_size if n_subcarriers is None else int(n_subcarriers)
        if np.any((pilot_subcarriers < 0) | (pilot_subcarriers >= N)):
            raise IndexError("Hay subportadoras piloto fuera de rango.")
        H_grid = np.zeros((M, N), dtype=np.complex128)
        H_grid[:, pilot_subcarriers] = H

    H_grid = _replace_invalid_with_zero(H_grid)

    if range_fft_size is None:
        range_fft_size = int(modulator_cfg.get("range_fft_size", params.range_fft_size))
    range_fft_size = int(range_fft_size)
    if range_fft_size < N:
        raise ValueError("range_fft_size debe ser >= número de subportadoras de la rejilla.")

    if apply_ifftshift:
        H_grid = np.fft.ifftshift(H_grid, axes=1)

    h_m_tau = np.fft.ifft(H_grid, n=range_fft_size, axis=1)
    h_tau_m = h_m_tau.T

    range_axis = compute_range_axis_from_cfg(
        modulator_cfg,
        n_range_bins=range_fft_size,
        distance_mode=distance_mode,
        c=c,
    )
    return h_tau_m, range_axis


def _slowtime_window(name: WindowName | None, length: int) -> np.ndarray:
    if name is None or name == "rect":
        return np.ones(length)
    if name == "hann":
        return np.hanning(length)
    if name == "hamming":
        return np.hamming(length)
    raise ValueError("Ventana no soportada. Usa None, 'rect', 'hann' o 'hamming'.")


def range_doppler_from_delay_time(
    h_tau_m: np.ndarray,
    *,
    modulator_cfg: dict[str, Any],
    range_axis: np.ndarray | None = None,
    doppler_fft_size: int | None = None,
    window_slowtime: WindowName | None = "hann",
    fftshift_doppler: bool = True,
    to_db: bool = False,
    power_floor: float = 1e-12,
    normalize: Literal["none", "window_energy", "num_samples"] = "none",
    distance_mode: DistanceMode = "monostatic",
    c: float = 299_792_458.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FFT sobre slow-time: h[tau,m] -> mapa rango-Doppler RD[tau,fD].

    Devuelve:
    - RD_power.shape = (N_range, N_doppler)
    - range_axis_out en metros
    - doppler_axis en Hz
    """
    h_tau_m = _ensure_complex_ndarray(h_tau_m, "h_tau_m")
    if h_tau_m.ndim != 2:
        raise ValueError(f"h_tau_m debe tener shape (N_range,M), recibido {h_tau_m.shape}")

    N_range, M = h_tau_m.shape
    params = get_ofdm_params(modulator_cfg)

    if range_axis is None:
        range_axis_out = compute_range_axis_from_cfg(
            modulator_cfg,
            n_range_bins=N_range,
            distance_mode=distance_mode,
            c=c,
        )
    else:
        range_axis_out = np.asarray(range_axis, dtype=np.float64)
        if range_axis_out.ndim != 1 or range_axis_out.shape[0] != N_range:
            raise ValueError(f"range_axis debe tener shape ({N_range},).")

    if doppler_fft_size is None:
        doppler_fft_size = int(modulator_cfg.get("doppler_fft_size", params.doppler_fft_size))
        if doppler_fft_size < M:
            doppler_fft_size = M
    doppler_fft_size = int(doppler_fft_size)
    if doppler_fft_size < M:
        raise ValueError("doppler_fft_size debe ser >= M.")

    w = _slowtime_window(window_slowtime, M)
    h_win = _replace_invalid_with_zero(h_tau_m) * w.reshape(1, M)

    RD_complex = np.fft.fft(h_win, n=doppler_fft_size, axis=1)
    if fftshift_doppler:
        RD_complex = np.fft.fftshift(RD_complex, axes=1)

    RD_power = np.abs(RD_complex) ** 2
    if normalize == "window_energy":
        RD_power = RD_power / max(np.sum(np.abs(w) ** 2), power_floor)
    elif normalize == "num_samples":
        RD_power = RD_power / max(M, 1)
    elif normalize != "none":
        raise ValueError("normalize debe ser 'none', 'window_energy' o 'num_samples'.")

    if to_db:
        RD_power = 10.0 * np.log10(np.maximum(RD_power, power_floor))

    doppler_axis = np.fft.fftfreq(doppler_fft_size, d=1.0 / params.fs_slow)
    if fftshift_doppler:
        doppler_axis = np.fft.fftshift(doppler_axis)

    return RD_power, range_axis_out, doppler_axis


def microdoppler_from_delay_time(
    h_tau_m: np.ndarray,
    *,
    modulator_cfg: dict[str, Any],
    range_axis: np.ndarray | None = None,
    delay_bin: int | None = None,
    select_bin: Literal["max_energy"] = "max_energy",
    min_range_m: float | None = None,
    max_range_m: float | None = None,
    nperseg: int = 64,
    noverlap: int = 48,
    nfft: int | None = None,
    window: str = "hann",
    fftshift_doppler: bool = True,
    to_db: bool = True,
    power_floor: float = 1e-12,
    detrend: bool | str = False,
    normalize: bool = False,
    distance_mode: DistanceMode = "monostatic",
    c: float = 299_792_458.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, np.ndarray]:
    """
    Rama micro-Doppler: selecciona un bin de retardo y calcula STFT en slow-time.

    Devuelve:
    - Smd.shape = (N_f, N_t)
    - f_md [Hz]
    - t_md [s]
    - selected_delay_bin
    - selected_range_m
    - slow_signal.shape = (M,)
    """
    h_tau_m = _ensure_complex_ndarray(h_tau_m, "h_tau_m")
    if h_tau_m.ndim != 2:
        raise ValueError(f"h_tau_m debe tener shape (N_range,M), recibido {h_tau_m.shape}")

    N_range, M = h_tau_m.shape
    if M < 2:
        raise ValueError("Se necesitan al menos dos muestras slow-time.")

    params = get_ofdm_params(modulator_cfg)

    if range_axis is None:
        range_axis = compute_range_axis_from_cfg(
            modulator_cfg,
            n_range_bins=N_range,
            distance_mode=distance_mode,
            c=c,
        )
    else:
        range_axis = np.asarray(range_axis, dtype=np.float64)
        if range_axis.ndim != 1 or range_axis.shape[0] != N_range:
            raise ValueError(f"range_axis debe tener shape ({N_range},).")

    if delay_bin is None:
        if select_bin != "max_energy":
            raise ValueError("select_bin solo admite 'max_energy'.")

        search_mask = np.ones(N_range, dtype=bool)
        if min_range_m is not None:
            search_mask &= range_axis >= min_range_m
        if max_range_m is not None:
            search_mask &= range_axis <= max_range_m
        if not np.any(search_mask):
            raise ValueError("La ventana de rango indicada no contiene ningún bin.")

        energy_per_bin = np.mean(np.abs(_replace_invalid_with_zero(h_tau_m)) ** 2, axis=1)
        energy_masked = np.where(search_mask, energy_per_bin, -np.inf)
        delay_bin = int(np.argmax(energy_masked))

    if not (0 <= delay_bin < N_range):
        raise IndexError(f"delay_bin={delay_bin} fuera de rango [0,{N_range - 1}].")

    selected_range_m = float(range_axis[delay_bin])
    slow_signal = _replace_invalid_with_zero(h_tau_m[delay_bin, :])

    if nfft is None:
        nfft = nperseg
    if nperseg > M:
        raise ValueError(f"nperseg={nperseg} no puede ser mayor que M={M}.")
    if nfft < nperseg:
        raise ValueError("nfft debe ser >= nperseg.")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap debe cumplir 0 <= noverlap < nperseg.")

    f_md, t_md, Zxx = stft(
        slow_signal,
        fs=params.fs_slow,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
        detrend=detrend,
    )

    Smd = np.abs(Zxx) ** 2
    if normalize:
        Smd = Smd / max(nperseg, 1)

    if fftshift_doppler:
        Smd = np.fft.fftshift(Smd, axes=0)
        f_md = np.fft.fftshift(f_md)

    if to_db:
        Smd = 10.0 * np.log10(np.maximum(Smd, power_floor))

    return Smd, f_md, t_md, delay_bin, selected_range_m, slow_signal


# =============================================================================
# Representación gráfica
# =============================================================================


def plot_ofdm_grid(
    grid: np.ndarray,
    t_sym: np.ndarray,
    f_sub: np.ndarray,
    *,
    title: str = "Rejilla OFDM",
    magnitude_db: bool = False,
    freq_unit: Literal["Hz", "kHz", "MHz"] = "MHz",
    time_unit: Literal["s", "ms", "us"] = "ms",
    fftshift_already_applied: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Representa módulo y fase de una rejilla OFDM compleja.
    """
    grid = _ensure_complex_ndarray(grid, "grid")
    t_sym = np.asarray(t_sym)
    f_sub = np.asarray(f_sub)

    if grid.ndim != 2:
        raise ValueError(f"grid debe tener shape (M,N), recibido {grid.shape}")
    M, N = grid.shape
    if len(t_sym) != M:
        raise ValueError(f"t_sym debe tener longitud {M}.")
    if len(f_sub) != N:
        raise ValueError(f"f_sub debe tener longitud {N}.")

    if fftshift_already_applied:
        grid_plot = grid
        f_plot = f_sub
    else:
        grid_plot = np.fft.fftshift(grid, axes=1)
        f_plot = np.fft.fftshift(f_sub)

    freq_scale = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6}[freq_unit]
    time_scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6}[time_unit]
    f_axis = f_plot / freq_scale
    t_axis = t_sym / time_scale

    mag = np.abs(grid_plot)
    if magnitude_db:
        mag = 20.0 * np.log10(np.maximum(mag, 1e-12))
        mag_label = "Módulo [dB]"
    else:
        mag_label = "Módulo"

    phase = np.angle(grid_plot)
    extent = [f_axis[0], f_axis[-1], t_axis[0], t_axis[-1]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)

    im0 = axes[0].imshow(mag, aspect="auto", origin="lower", interpolation="nearest", extent=extent)
    axes[0].set_title(f"{title} - Módulo")
    axes[0].set_ylabel(f"Tiempo [{time_unit}]")
    cbar0 = fig.colorbar(im0, ax=axes[0])
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
    axes[1].set_title(f"{title} - Fase")
    axes[1].set_xlabel(f"Frecuencia de subportadora [{freq_unit}]")
    axes[1].set_ylabel(f"Tiempo [{time_unit}]")
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label("Fase [rad]")
    cbar1.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar1.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    return fig, axes


def plot_rd_and_microdoppler(
    RD: np.ndarray,
    range_axis: np.ndarray,
    doppler_axis: np.ndarray,
    Smd: np.ndarray,
    f_md: np.ndarray,
    t_md: np.ndarray,
    *,
    delay_bin: int | None = None,
    selected_range_m: float | None = None,
    rd_in_db: bool = True,
    microdoppler_in_db: bool = True,
    range_limits: tuple[float, float] | None = None,
    doppler_limits: tuple[float, float] | None = None,
    md_doppler_limits: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (14, 10),
    cmap_rd: str = "viridis",
    cmap_md: str = "viridis",
    show_colorbars: bool = True,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Representa simultáneamente el mapa rango-Doppler y el espectrograma micro-Doppler.
    """
    RD = np.asarray(RD)
    range_axis = np.asarray(range_axis)
    doppler_axis = np.asarray(doppler_axis)
    Smd = np.asarray(Smd)
    f_md = np.asarray(f_md)
    t_md = np.asarray(t_md)

    if RD.ndim != 2:
        raise ValueError(f"RD debe tener shape (N_range,N_doppler), recibido {RD.shape}")
    if Smd.ndim != 2:
        raise ValueError(f"Smd debe tener shape (N_f,N_t), recibido {Smd.shape}")
    if range_axis.shape != (RD.shape[0],):
        raise ValueError(f"range_axis debe tener shape ({RD.shape[0]},).")
    if doppler_axis.shape != (RD.shape[1],):
        raise ValueError(f"doppler_axis debe tener shape ({RD.shape[1]},).")
    if f_md.shape != (Smd.shape[0],):
        raise ValueError(f"f_md debe tener shape ({Smd.shape[0]},).")
    if t_md.shape != (Smd.shape[1],):
        raise ValueError(f"t_md debe tener shape ({Smd.shape[1]},).")

    if delay_bin is not None:
        if not (0 <= delay_bin < len(range_axis)):
            raise IndexError(f"delay_bin={delay_bin} fuera de rango.")
        dominant_distance = float(range_axis[delay_bin]) if selected_range_m is None else float(selected_range_m)
    else:
        dominant_distance = None

    fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)

    rd_extent = [doppler_axis[0], doppler_axis[-1], range_axis[0], range_axis[-1]]
    im0 = axes[0].imshow(RD, origin="lower", aspect="auto", extent=rd_extent, cmap=cmap_rd)
    axes[0].set_title("Mapa rango-Doppler")
    axes[0].set_xlabel("Frecuencia Doppler [Hz]")
    axes[0].set_ylabel("Rango [m]")

    if delay_bin is not None and dominant_distance is not None:
        axes[0].axhline(
            dominant_distance,
            linestyle="--",
            linewidth=1.5,
            color="white",
            label=f"Bin dominante = {delay_bin}, R = {dominant_distance:.3f} m",
        )
        axes[0].legend(loc="upper right")

    if doppler_limits is not None:
        axes[0].set_xlim(*doppler_limits)
    if range_limits is not None:
        axes[0].set_ylim(*range_limits)
    if show_colorbars:
        cbar0 = fig.colorbar(im0, ax=axes[0])
        cbar0.set_label("Potencia [dB]" if rd_in_db else "Potencia")

    md_extent = [t_md[0], t_md[-1], f_md[0], f_md[-1]]
    im1 = axes[1].imshow(Smd, origin="lower", aspect="auto", extent=md_extent, cmap=cmap_md)
    if delay_bin is not None and dominant_distance is not None:
        axes[1].set_title(f"Espectrograma micro-Doppler (bin = {delay_bin}, R = {dominant_distance:.3f} m)")
    else:
        axes[1].set_title("Espectrograma micro-Doppler")
    axes[1].set_xlabel("Tiempo [s]")
    axes[1].set_ylabel("Frecuencia Doppler [Hz]")

    if md_doppler_limits is not None:
        axes[1].set_ylim(*md_doppler_limits)
    if show_colorbars:
        cbar1 = fig.colorbar(im1, ax=axes[1])
        cbar1.set_label("Potencia [dB]" if microdoppler_in_db else "Potencia")

    if title is not None:
        fig.suptitle(title)

    return fig, axes


# =============================================================================
# Ejemplo de flujo de alto nivel
# =============================================================================


def openisac_maps_from_channel(
    H: np.ndarray,
    *,
    modulator_cfg: dict[str, Any],
    pilot_subcarriers: np.ndarray | None = None,
    distance_mode: DistanceMode = "monostatic",
    min_range_m: float | None = None,
    max_range_m: float | None = None,
    rd_to_db: bool = True,
    md_to_db: bool = True,
    md_nperseg: int = 64,
    md_noverlap: int = 48,
    md_nfft: int | None = None,
) -> dict[str, Any]:
    """
    Atajo: H -> h[tau,m] -> mapa RD + espectrograma micro-Doppler.
    """
    h_tau_m, range_axis = delay_time_ifft_from_H(
        H,
        modulator_cfg=modulator_cfg,
        pilot_subcarriers=pilot_subcarriers,
        distance_mode=distance_mode,
    )

    RD, range_axis, doppler_axis = range_doppler_from_delay_time(
        h_tau_m,
        modulator_cfg=modulator_cfg,
        range_axis=range_axis,
        to_db=rd_to_db,
        distance_mode=distance_mode,
    )

    Smd, f_md, t_md, delay_bin, selected_range_m, slow_signal = microdoppler_from_delay_time(
        h_tau_m,
        modulator_cfg=modulator_cfg,
        range_axis=range_axis,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
        nperseg=md_nperseg,
        noverlap=md_noverlap,
        nfft=md_nfft,
        to_db=md_to_db,
        distance_mode=distance_mode,
    )

    return {
        "h_tau_m": h_tau_m,
        "range_axis": range_axis,
        "RD": RD,
        "doppler_axis": doppler_axis,
        "Smd": Smd,
        "f_md": f_md,
        "t_md": t_md,
        "delay_bin": delay_bin,
        "selected_range_m": selected_range_m,
        "slow_signal": slow_signal,
    }


def openisac_maps_from_iq_bins(
    tx_bin_path: str | Path,
    rx_bin_path: str | Path,
    yaml_path: str | Path,
    *,
    tx_storage_format: IQStorageFormat = "auto",
    rx_storage_format: IQStorageFormat = "auto",
    n_symbols: int | None = None,
    start_sample_tx: int = 0,
    start_sample_rx: int = 0,
    fftshift: bool = False,
    normalize_fft: bool = False,
    apply_ifftshift: bool = False,
    distance_mode: str = "monostatic",
    window_slowtime: str | None = "hann",
    to_db_rd: bool = True,
    to_db_md: bool = True,
    min_range_m: float | None = None,
    max_range_m: float | None = None,
    delay_bin: int | None = None,
    select_bin: str = "max_energy",
    md_nperseg: int = 64,
    md_noverlap: int = 48,
    md_nfft: int | None = None,
    md_window: str = "hann",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Pipeline alto nivel desde dos archivos IQ binarios TX/RX hasta:

    - mapa rango-Doppler
    - espectrograma micro-Doppler

    Flujo:
        iq_tx.bin, iq_rx.bin, Modulator.yaml
            -> lectura IQ
            -> CP removal + FFT
            -> X[m,k], Y[m,k]
            -> H[m,k] = Y[m,k] / X[m,k]
            -> IFFT_k -> h[tau,m]
            -> FFT_m -> RD[range,Doppler]
            -> selección delay_bin + STFT -> micro-Doppler

    Returns
    -------
    out : dict
        Diccionario con:
        - cfg
        - params
        - tx_iq, rx_iq
        - X, Y
        - H, valid_mask
        - h_tau_m
        - range_axis
        - RD, doppler_axis
        - Smd, f_md, t_md
        - delay_bin
        - selected_range_m
        - slow_signal
    """

    cfg = load_modulator_yaml(yaml_path)
    params = get_ofdm_params(cfg)

    # ------------------------------------------------------------
    # 1) Lectura IQ TX/RX
    # ------------------------------------------------------------
    tx_iq = read_usrp_iq_bin(
        tx_bin_path,
        yaml_path,
        storage_format=tx_storage_format,
        trim_to_complete_frames=False,
        return_time_axis=False,
        verbose=verbose,
    )

    rx_iq = read_usrp_iq_bin(
        rx_bin_path,
        yaml_path,
        storage_format=rx_storage_format,
        trim_to_complete_frames=False,
        return_time_axis=False,
        verbose=verbose,
    )

    # ------------------------------------------------------------
    # 2) Demodulación OFDM: CP removal + FFT
    # ------------------------------------------------------------
    X = demodulate_ofdm_iq(
        tx_iq,
        yaml_path,
        n_symbols=n_symbols,
        start_sample=start_sample_tx,
        fftshift=fftshift,
        normalize_fft=normalize_fft,
        return_axes=False,
        verbose=verbose,
    )

    Y = demodulate_ofdm_iq(
        rx_iq,
        yaml_path,
        n_symbols=n_symbols,
        start_sample=start_sample_rx,
        fftshift=fftshift,
        normalize_fft=normalize_fft,
        return_axes=False,
        verbose=verbose,
    )

    # Si por cualquier motivo se extraen números distintos de símbolos,
    # recortamos a la longitud común.
    M_common = min(X.shape[0], Y.shape[0])

    if M_common <= 0:
        raise ValueError("No hay símbolos OFDM comunes entre TX y RX.")

    X = X[:M_common, :]
    Y = Y[:M_common, :]

    # ------------------------------------------------------------
    # 3) Estimación de canal H = Y/X
    # ------------------------------------------------------------
    H, valid_mask = estimate_channel_grid(
        X,
        Y,
        return_mask=True,
        verbose=verbose,
    )

    # ------------------------------------------------------------
    # 4) Dominio retardo-tiempo: IFFT sobre subportadoras
    # ------------------------------------------------------------
    h_tau_m, range_axis = delay_time_ifft_from_H(
        H,
        modulator_cfg=cfg,
        range_fft_size=params.range_fft_size,
        apply_ifftshift=apply_ifftshift,
        distance_mode=distance_mode,
    )

    # ------------------------------------------------------------
    # 5) Rama rango-Doppler
    # ------------------------------------------------------------
    RD, range_axis_rd, doppler_axis = range_doppler_from_delay_time(
        h_tau_m,
        modulator_cfg=cfg,
        range_axis=range_axis,
        doppler_fft_size=params.doppler_fft_size,
        window_slowtime=window_slowtime,
        fftshift_doppler=True,
        to_db=to_db_rd,
        distance_mode=distance_mode,
    )

    # ------------------------------------------------------------
    # 6) Rama micro-Doppler
    # ------------------------------------------------------------
    md_out = microdoppler_from_delay_time(
    h_tau_m,
    modulator_cfg=cfg,
    range_axis=range_axis,
    delay_bin=delay_bin,
    select_bin=select_bin,
    min_range_m=min_range_m,
    max_range_m=max_range_m,
    nperseg=md_nperseg,
    noverlap=md_noverlap,
    nfft=md_nfft,
    window=md_window,
    fftshift_doppler=True,
    to_db=to_db_md,
    )

    if len(md_out) == 5:
        Smd, f_md, t_md, delay_bin_used, slow_signal = md_out
        selected_range_m = float(range_axis[delay_bin_used])

    elif len(md_out) == 6:
        Smd, f_md, t_md, delay_bin_used, selected_range_m, slow_signal = md_out

    else:
        raise ValueError(
            f"microdoppler_from_delay_time devolvió {len(md_out)} valores; "
            "se esperaban 5 o 6."
        )

    if verbose:
        print("Pipeline OpenISAC desde IQ completado")
        print(f"  Shape X: {X.shape}")
        print(f"  Shape Y: {Y.shape}")
        print(f"  Shape H: {H.shape}")
        print(f"  Shape h_tau_m: {h_tau_m.shape}")
        print(f"  Shape RD: {RD.shape}")
        print(f"  Shape Smd: {Smd.shape}")
        print(f"  Delay bin seleccionado: {delay_bin_used}")
        print(f"  Rango seleccionado: {selected_range_m:.6f} m")

    return {
        "cfg": cfg,
        "params": params,
        "tx_iq": tx_iq,
        "rx_iq": rx_iq,
        "X": X,
        "Y": Y,
        "H": H,
        "valid_mask": valid_mask,
        "h_tau_m": h_tau_m,
        "range_axis": range_axis_rd,
        "RD": RD,
        "doppler_axis": doppler_axis,
        "Smd": Smd,
        "f_md": f_md,
        "t_md": t_md,
        "delay_bin": delay_bin_used,
        "selected_range_m": selected_range_m,
        "slow_signal": slow_signal,
    }