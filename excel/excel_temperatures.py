"""
excel_temperatures.py
---------------------
Recorre automáticamente todas las carpetas de temperatura que existan en data/
(p.ej. data/40/, data/55/, data/70/ ...).

Dentro de cada carpeta busca pares TX/RX:
    iq_tx_1.bin  <->  iq_rx_1.bin
    iq_tx_2.bin  <->  iq_rx_2.bin
    ...
    iq_tx.bin    <->  iq_rx.bin        (sin número, también aceptado)

Para cada par calcula la matriz de canal H y extrae las características
definidas en channel_features_David.py. El resultado se guarda en un Excel
con el mismo formato que Pearson_David.xlsx.

Solo se procesan las primeras MAX_PAIRS_PER_TEMP medidas por carpeta de
temperatura (por defecto 100), ordenadas numéricamente.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuración ─────────────────────────────────────────────────────────────
MAX_PAIRS_PER_TEMP = 100          # Máximo de medidas por temperatura

# ── Rutas base ────────────────────────────────────────────────────────────────
EXCEL_DIR  = Path(__file__).resolve().parent         
REPO_ROOT   = EXCEL_DIR.parent 
ISAC_DIR    = REPO_ROOT.parent                    
DATA_DIR    = ISAC_DIR / "BBDD"
MEDIDAS_DIR = DATA_DIR 
YAML_PATH   = REPO_ROOT / "data" / "Modulator.yaml"
OUTPUT_PATH = EXCEL_DIR / "features_temperatures.xlsx"
SCRIPTS_DIR = REPO_ROOT / "isac"

# Añade Hugo/ al path para que funcionen los mismos imports que en excel.py
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from channel_estimator import compute_channel_matrix_from_iq_paths  # noqa: E402
from channel_features_David import extract_channel_matrix_features    # noqa: E402


# ── Descubrimiento ────────────────────────────────────────────────────────────

def _is_temperature_folder(name: str) -> bool:
    """True si el nombre de carpeta es un número (entero o decimal)."""
    return bool(re.fullmatch(r"\d+(\.\d+)?", name))


def _discover_pairs(temp_dir: Path, max_pairs: int = MAX_PAIRS_PER_TEMP) -> list[dict]:
    """
    Busca los primeros ``max_pairs`` pares TX/RX dentro de una carpeta de
    temperatura, ordenados numéricamente.

    Acepta:
        iq_tx_<N>.bin  /  iq_rx_<N>.bin    (pares numerados)
        iq_tx.bin      /  iq_rx.bin         (par sin número, índice "0")

    Devuelve lista de dicts con: index, tx_path, rx_path.
    """
    pairs = []

    # Ordenar numéricamente para garantizar 1, 2, 3 … en lugar de 1, 10, 11 …
    def _numeric_key(p: Path) -> int:
        m = re.search(r"iq_tx_(\d+)\.bin$", p.name)
        return int(m.group(1)) if m else -1

    for tx in sorted(temp_dir.glob("iq_tx_*.bin"), key=_numeric_key):
        m = re.search(r"iq_tx_(\d+)\.bin$", tx.name)
        if not m:
            continue
        idx = m.group(1)
        rx = temp_dir / f"iq_rx_{idx}.bin"
        if rx.exists():
            pairs.append({"index": idx, "tx_path": tx, "rx_path": rx})

    tx_plain = temp_dir / "iq_tx.bin"
    rx_plain = temp_dir / "iq_rx.bin"
    if tx_plain.exists() and rx_plain.exists():
        pairs.append({"index": "0", "tx_path": tx_plain, "rx_path": rx_plain})

    return pairs[:max_pairs]


# ── Filtrado de escalares ─────────────────────────────────────────────────────

def _keep_scalars(features: dict) -> dict:
    return {
        k: float(v)
        for k, v in features.items()
        if not isinstance(v, np.ndarray) and np.isscalar(v)
    }


# ── Loop principal ────────────────────────────────────────────────────────────

def main() -> None:
    if not MEDIDAS_DIR.exists():
        raise FileNotFoundError(f"No se encuentra la carpeta de medidas: {MEDIDAS_DIR}")

    if not YAML_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el YAML: {YAML_PATH}")

    temp_dirs = sorted(
        (d for d in MEDIDAS_DIR.iterdir() if d.is_dir() and _is_temperature_folder(d.name)),
        key=lambda d: float(d.name),
    )

    if not temp_dirs:
        print(f"No se encontraron carpetas de temperatura en {MEDIDAS_DIR}")
        print("Las carpetas deben tener nombres numéricos (p.ej. 40, 55, 70).")
        return

    print(f"Carpetas de temperatura encontradas ({len(temp_dirs)}): "
          f"{[d.name for d in temp_dirs]}")
    print(f"YAML           : {YAML_PATH}")
    print(f"Límite medidas : {MAX_PAIRS_PER_TEMP} por temperatura")
    print()

    rows = []

    for temp_dir in temp_dirs:
        temperature = temp_dir.name
        pairs = _discover_pairs(temp_dir)

        if not pairs:
            print(f"[{temperature}°C] Sin pares TX/RX — se omite.")
            continue

        print(f"[{temperature}°C] {len(pairs)} par(es) a procesar "
              f"(de un máximo de {MAX_PAIRS_PER_TEMP})")

        for pair in pairs:
            sample_id = f"temp_{temperature}_sample_{pair['index']}"
            tx_path   = pair["tx_path"]
            rx_path   = pair["rx_path"]

            print(f"  Procesando {sample_id} ...", end=" ", flush=True)

            try:
                H = compute_channel_matrix_from_iq_paths(
                    tx_path,
                    rx_path,
                    YAML_PATH,
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
                features = _keep_scalars(features)

                row = {
                    "id":          sample_id,
                    "temperature": float(temperature),
                    "sample_idx":  pair["index"],
                    "tx_path":     str(tx_path),
                    "rx_path":     str(rx_path),
                    "status":      "ok",
                }
                row.update(features)
                print("OK")

            except Exception as exc:
                row = {
                    "id":            sample_id,
                    "temperature":   float(temperature),
                    "sample_idx":    pair["index"],
                    "tx_path":       str(tx_path),
                    "rx_path":       str(rx_path),
                    "status":        "error",
                    "error_message": str(exc),
                }
                print(f"ERROR: {exc}")

            rows.append(row)

    if not rows:
        print("No se generaron filas. Revisa la estructura de datos.")
        return

    df = pd.DataFrame(rows)

    meta_cols = ["id", "temperature", "sample_idx", "tx_path", "rx_path", "status"]
    if "error_message" in df.columns:
        meta_cols.append("error_message")
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feat_cols]

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="caracteristicas_H", index=False)

        ws = writer.sheets["caracteristicas_H"]
        for col_cells in ws.columns:
            max_len = max(
                (len(str(cell.value)) for cell in col_cells if cell.value is not None),
                default=0,
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)

    print()
    print("=" * 60)
    print(f"Excel generado      : {OUTPUT_PATH.resolve()}")
    print(f"Muestras procesadas : {len(rows)}")
    print(f"Características     : {len(feat_cols)}")
    print(f"Shape tabla         : {df.shape}")


if __name__ == "__main__":
    main()
