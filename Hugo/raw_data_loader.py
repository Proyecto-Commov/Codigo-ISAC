# raw_data_loader.py
from pathlib import Path
from typing import Any, Dict
import numpy as np

from signal_types import SignalBlock


class RawDataLoader:
    """
    Cargador genérico de datos.
    Soporta:
      - .npy
      - .npz
      - .dat / .cfile (complex64 intercalado I/Q)
    Puedes ampliarlo a .h5/.mat si lo necesitas.
    """

    def load(self, filepath: str, source_type: str = "generic", **kwargs) -> SignalBlock:
        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix == ".npy":
            data = np.load(path, allow_pickle=True)
        elif suffix == ".npz":
            npz = np.load(path, allow_pickle=True)
            if "data" in npz:
                data = npz["data"]
            else:
                # toma el primer array
                first_key = list(npz.keys())[0]
                data = npz[first_key]
        elif suffix in {".dat", ".cfile"}:
            data = self._load_complex_binary(path, dtype=np.complex64)
        else:
            raise ValueError(f"Formato no soportado: {suffix}")

        metadata: Dict[str, Any] = dict(kwargs)
        metadata["filepath"] = str(path)
        return SignalBlock(source_type=source_type, data=data, metadata=metadata)

    @staticmethod
    def _load_complex_binary(path: Path, dtype=np.complex64) -> np.ndarray:
        """
        Asume que el fichero ya contiene complejos (por ejemplo complex64).
        Si tu .dat viene como float intercalado I,Q, habría que adaptar esto.
        """
        return np.fromfile(path, dtype=dtype)