# ofdm_frame_parser.py
import numpy as np

from config import PipelineConfig
from signal_types import SignalBlock, OFDMGrid


class OFDMFrameParser:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.nfft = cfg.n_subcarriers
        self.cp = cfg.cp_len
        self.nsym = cfg.n_symbols_per_frame
        self.samples_per_symbol = self.nfft + self.cp
        self.samples_per_frame = self.nsym * self.samples_per_symbol

    def parse(self, block: SignalBlock) -> OFDMGrid:
        kind = block.metadata.get("data_kind", None)

        if kind == "ofdm_grid":
            grid = np.asarray(block.data)
            self._validate_grid(grid)
            return OFDMGrid(grid=grid, metadata=dict(block.metadata))

        if kind != "iq":
            raise ValueError("Tipo de entrada no reconocido por OFDMFrameParser.")

        iq = np.asarray(block.data)
        if not self.cfg.assume_frame_aligned:
            raise NotImplementedError(
                "La sincronización automática de tramas no está implementada en esta versión."
            )

        n_frames = len(iq) // self.samples_per_frame
        if n_frames < 1:
            raise ValueError("No hay suficientes muestras para formar una trama OFDM.")

        iq = iq[: n_frames * self.samples_per_frame]
        frames = iq.reshape(n_frames, self.nsym, self.samples_per_symbol)

        # quitar CP
        no_cp = frames[:, :, self.cp:]

        # FFT por símbolo
        grid = np.fft.fft(no_cp, axis=-1)

        self._validate_grid(grid)
        metadata = dict(block.metadata)
        metadata["parsed_from_iq"] = True
        return OFDMGrid(grid=grid, metadata=metadata)

    def _validate_grid(self, grid: np.ndarray) -> None:
        if grid.ndim != 3:
            raise ValueError("La rejilla OFDM debe tener forma [n_frames, n_symbols, n_subcarriers].")
        if grid.shape[1] != self.nsym:
            raise ValueError(
                f"Se esperaban {self.nsym} símbolos por trama y llegaron {grid.shape[1]}."
            )
        if grid.shape[2] != self.nfft:
            raise ValueError(
                f"Se esperaban {self.nfft} subportadoras y llegaron {grid.shape[2]}."
            )