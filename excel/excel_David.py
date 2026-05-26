from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Rutas clave derivadas de la posicion de este archivo (funciona en Windows y Linux)
_here      = Path(__file__).resolve().parent          # Hugo/excel/
_hugo      = _here.parent                             # Hugo/
_repo_root = _hugo.parent                             # Codigo-ISAC/
_data_root = _repo_root / "data"                      # Codigo-ISAC/data/

for _p in [str(_hugo), str(_repo_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from channel_estimator import compute_channel_matrix_from_iq_paths
from channel_features_David import extract_channel_matrix_features


# =============================================================================
# Configuracion
# =============================================================================

yaml_path = str(_data_root / "Modulator.yaml")

output_excel_path = str(_here / "Pearson_David.xlsx")

# Mapeo id -> etiqueta numerica de temperatura
#   frio=0  templado=1  caliente=2  vacio=NaN
def _hot_state(measure_id: str):
    s = measure_id.lower()
    if "caliente" in s:
        return 2
    if "templado" in s:
        return 1
    if "frio" in s or "frío" in s:
        return 0
    return float("nan")   # vacio u otro


# =============================================================================
# Pares de senales TX / RX
# =============================================================================

def _pair(label: str, tx: str, rx: str) -> dict:
    return {
        "id":      label,
        "tx_path": str(_data_root / tx),
        "rx_path": str(_data_root / rx),
    }


signal_pairs = [
    _pair("vacio",            "vacio/iq_tx.bin",               "vacio/iq_rx.bin"),
    _pair("cristal caliente 1", "cristal/caliente/iq_tx_1.bin", "cristal/caliente/iq_rx_1.bin"),
    _pair("cristal caliente 2", "cristal/caliente/iq_tx_2.bin", "cristal/caliente/iq_rx_2.bin"),
    _pair("carton caliente 1",  "carton/caliente/iq_tx_1.bin",  "carton/caliente/iq_rx_1.bin"),
    _pair("carton caliente 2",  "carton/caliente/iq_tx_2.bin",  "carton/caliente/iq_rx_2.bin"),
    _pair("plastico caliente 1","plastico/caliente/iq_tx_1.bin","plastico/caliente/iq_rx_1.bin"),
    _pair("plastico caliente 2","plastico/caliente/iq_tx_2.bin","plastico/caliente/iq_rx_2.bin"),
    _pair("cristal templado 1", "cristal/templado/iq_tx_1.bin", "cristal/templado/iq_rx_1.bin"),
    _pair("cristal templado 2", "cristal/templado/iq_tx_2.bin", "cristal/templado/iq_rx_2.bin"),
    _pair("carton templado 1",  "carton/templado/iq_tx_1.bin",  "carton/templado/iq_rx_1.bin"),
    _pair("carton templado 2",  "carton/templado/iq_tx_2.bin",  "carton/templado/iq_rx_2.bin"),
    _pair("plastico templado 1","plastico/templado/iq_tx_1.bin","plastico/templado/iq_rx_1.bin"),
    _pair("plastico templado 2","plastico/templado/iq_tx_2.bin","plastico/templado/iq_rx_2.bin"),
    _pair("cristal frio 1",     "cristal/frio/iq_tx_1.bin",     "cristal/frio/iq_rx_1.bin"),
    _pair("cristal frio 2",     "cristal/frio/iq_tx_2.bin",     "cristal/frio/iq_rx_2.bin"),
    _pair("carton frio 1",      "carton/frio/iq_tx_1.bin",      "carton/frio/iq_rx_1.bin"),
    _pair("carton frio 2",      "carton/frio/iq_tx_2.bin",      "carton/frio/iq_rx_2.bin"),
    _pair("plastico frio 1",    "plastico/frio/iq_tx_1.bin",    "plastico/frio/iq_rx_1.bin"),
    _pair("plastico frio 2",    "plastico/frio/iq_tx_2.bin",    "plastico/frio/iq_rx_2.bin"),
]


# =============================================================================
# Calculo de caracteristicas
# =============================================================================

rows = []

for pair in signal_pairs:
    measure_id = pair["id"]
    tx_path    = pair["tx_path"]
    rx_path    = pair["rx_path"]

    print("=" * 80)
    print(f"Procesando: {measure_id}")

    try:
        H = compute_channel_matrix_from_iq_paths(
            tx_path,
            rx_path,
            yaml_path,
            n_symbols=100,
            storage_format_tx="auto",
            storage_format_rx="auto",
            fftshift=False,
            normalize_fft=False,
            output_order="mk",
            return_aux=False,
            verbose=False,
        )

        features = extract_channel_matrix_features(H, input_order="mk")

        # Filtrar solo escalares
        features = {
            k: float(v)
            for k, v in features.items()
            if np.isscalar(v) and not isinstance(v, np.ndarray)
        }

        row = {
            "id":       measure_id,
            "tx_path":  tx_path,
            "rx_path":  rx_path,
            "status":   "ok",
            "hot_state": _hot_state(measure_id),
        }
        row.update(features)

    except Exception as e:
        row = {
            "id":            measure_id,
            "tx_path":       tx_path,
            "rx_path":       rx_path,
            "status":        "error",
            "error_message": str(e),
            "hot_state":     _hot_state(measure_id),
        }
        print(f"  ERROR: {e}")

    rows.append(row)


# =============================================================================
# Exportar a Excel
# =============================================================================

df = pd.DataFrame(rows)

# Orden de columnas: metadatos primero, hot_state al final
meta_cols = ["id", "tx_path", "rx_path", "status"]
if "error_message" in df.columns:
    meta_cols.append("error_message")

_new_vars = {
    "mag_skewness", "pdp_mean_delay", "pdp_rms_delay_spread", "pdp_los_ratio",
    "svd_sigma_ratio", "dH_dt_mean", "dH_dt_max", "doppler_spread",
}

feature_cols_orig = [
    c for c in df.columns
    if c not in meta_cols and c != "hot_state" and c not in _new_vars
]
feature_cols_new = [
    c for c in df.columns
    if c in _new_vars
]

df = df[meta_cols + feature_cols_orig + feature_cols_new + ["hot_state"]]

with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="caracteristicas_H", index=False)

    ws = writer.sheets["caracteristicas_H"]
    for col_cells in ws.columns:
        max_len = max(
            (len(str(cell.value)) for cell in col_cells if cell.value is not None),
            default=0,
        )
        ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)

print("=" * 80)
print("Excel generado correctamente")
print(f"Ruta: {Path(output_excel_path).resolve()}")
print(f"Shape: {df.shape}  ({df.shape[0]} filas x {df.shape[1]} columnas)")
print(f"Nuevas variables anadidas: mag_skewness, pdp_mean_delay, pdp_rms_delay_spread,")
print(f"  pdp_los_ratio, svd_sigma_ratio, dH_dt_mean, dH_dt_max,")
print(f"  doppler_centroid, doppler_spread")
