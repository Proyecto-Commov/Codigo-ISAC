# format_adapter.py
import numpy as np

from Hugo.prototype.signal_types import SignalBlock


class FormatAdapter:
    """
    Normaliza el formato de entrada.

    Convenciones:
      - Si data.ndim == 3: se interpreta como rejilla OFDM
        [n_frames, n_symbols, n_subcarriers]
      - Si data.ndim == 1: se interpreta como IQ crudo complejo
    """

    def adapt(self, block: SignalBlock) -> SignalBlock:
        data = np.asarray(block.data)

        if data.ndim == 3:
            block.metadata["data_kind"] = "ofdm_grid"
        elif data.ndim == 1:
            if not np.iscomplexobj(data):
                raise ValueError("IQ crudo debe ser complejo.")
            block.metadata["data_kind"] = "iq"
        else:
            raise ValueError(
                f"Dimensión de datos no soportada: {data.ndim}. "
                "Se esperaba 1D (IQ) o 3D (rejilla OFDM)."
            )

        block.data = data
        return block