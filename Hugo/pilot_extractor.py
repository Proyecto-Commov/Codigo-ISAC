# pilot_extractor.py
import numpy as np

from config import PipelineConfig
from signal_types import OFDMGrid, PilotObservations


def zadoff_chu_seq(length: int, root: int) -> np.ndarray:
    """
    Secuencia Zadoff-Chu para longitud N.
    Implementación simple:
      z[n] = exp(-j*pi*q*n*(n+1)/N) si N es impar
      z[n] = exp(-j*pi*q*n^2/N)      si N es par
    """
    n = np.arange(length)
    if length % 2 == 0:
        z = np.exp(-1j * np.pi * root * (n ** 2) / length)
    else:
        z = np.exp(-1j * np.pi * root * n * (n + 1) / length)
    return z.astype(np.complex128)


class PilotExtractor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.n = cfg.n_subcarriers
        self.msync = cfg.pilot.msync
        self.pilot_idx = cfg.pilot.pilot_indices(self.n)
        self.zc = zadoff_chu_seq(self.n, cfg.pilot.zc_root)

    def extract(self, ofdm: OFDMGrid) -> PilotObservations:
        grid = ofdm.grid  # [F, M, N]
        n_frames, n_symbols, n_subcarriers = grid.shape

        if self.msync >= n_symbols:
            raise ValueError("msync está fuera del número de símbolos por trama.")

        # Símbolos efectivos para sensing:
        # tomamos todos, pero en msync usamos full-band;
        # en los demás, solo los pilotos fijos.
        symbol_indices = np.arange(n_symbols, dtype=int)

        y_list = []
        x_list = []

        for m in symbol_indices:
            if m == self.msync and self.cfg.pilot.use_full_sync_symbol:
                y_m = grid[:, m, :]               # [F, N]
                x_m = np.broadcast_to(self.zc, y_m.shape)
            else:
                y_m = grid[:, m, self.pilot_idx]  # [F, P]
                x_m = np.broadcast_to(self.zc[self.pilot_idx], y_m.shape)

            y_list.append(y_m)
            x_list.append(x_m)

        # Queremos [F, M_eff, P_var]
        # Como msync tiene N y el resto P, hay tamaños distintos.
        # Para simplificar E3.1, solo usamos la parte común de pilotos fijos
        # en todos los símbolos, excluyendo el full-band de msync del pipeline Doppler.
        sensing_symbol_indices = np.array(
            [m for m in symbol_indices if m != self.msync], dtype=int
        )

        y_pilots = grid[:, sensing_symbol_indices][:, :, self.pilot_idx]          # [F, M-1, P]
        x_pilots = np.broadcast_to(self.zc[self.pilot_idx], y_pilots.shape)

        metadata = dict(ofdm.metadata)
        metadata["msync"] = self.msync
        metadata["pilot_pattern"] = "openisac_fixed_subcarriers"
        return PilotObservations(
            y_pilots=y_pilots,
            x_pilots=x_pilots,
            pilot_indices=self.pilot_idx,
            symbol_indices=sensing_symbol_indices,
            metadata=metadata,
        )