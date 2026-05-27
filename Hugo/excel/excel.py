from pathlib import Path
import numpy as np
import pandas as pd

from channel_estimator import compute_channel_matrix_from_iq_paths
from Hugo.prototype3.channel_features import extract_channel_matrix_features


# =============================================================================
# Configuración
# =============================================================================

yaml_path = "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/Modulator.yaml"

output_excel_path = "/home/hugo/Escritorio/ISAC/Codigo-ISAC/Hugo/pearson.xlsx"

# -------------------------------------------------------------------------
# Define aquí tus 12 pares de señales TX/RX
# -------------------------------------------------------------------------

signal_pairs = [
    {
        "id": "vacio",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/vacio/iq_tx.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/vacio/iq_rx.bin",
    },
    {
        "id": "cristal caliente 1",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/cristal/caliente/iq_tx_1.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/cristal/caliente/iq_rx_1.bin",
    },
    {
        "id": "cristal caliente 2",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/cristal/caliente/iq_tx_2.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/cristal/caliente/iq_rx_2.bin",
    },
    {
        "id": "cartón caliente 1",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/carton/caliente/iq_tx_1.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/carton/caliente/iq_rx_1.bin",
    },
    {
        "id": "cartón caliente 2",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/carton/caliente/iq_tx_2.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/carton/caliente/iq_rx_2.bin",
    },
    {
        "id": "plástico caliente 1",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/plastico/caliente/iq_tx_1.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/plastico/caliente/iq_rx_1.bin",
    },
    {
        "id": "plástico caliente 2",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/plastico/caliente/iq_tx_2.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/plastico/caliente/iq_rx_2.bin",
    },
    {
        "id": "cristal templado 1",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/cristal/templado/iq_tx_1.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/cristal/templado/iq_rx_1.bin",
    },
    {
        "id": "cristal templado 2",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/cristal/templado/iq_tx_2.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/cristal/templado/iq_rx_2.bin",
    },
    {
        "id": "cartón templado 1",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/carton/templado/iq_tx_1.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/carton/templado/iq_rx_1.bin",
    },
    {
        "id": "cartón templado 2",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/carton/templado/iq_tx_2.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/carton/templado/iq_rx_2.bin",
    },
    {
        "id": "plástico templado 1",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/plastico/templado/iq_tx_1.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/plastico/templado/iq_rx_1.bin",
    },
    {
        "id": "plástico templado 2",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/plastico/templado/iq_tx_2.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/plastico/templado/iq_rx_2.bin",
    },
    {
        "id": "cristal frío 1",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/cristal/frio/iq_tx_1.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/cristal/frio/iq_rx_1.bin",
    },
    {
        "id": "cristal frío 2",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/cristal/frio/iq_tx_2.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/cristal/frio/iq_rx_2.bin",
    },
    {
        "id": "cartón frío 1",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/carton/frio/iq_tx_1.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/carton/frio/iq_rx_1.bin",
    },
    {
        "id": "cartón frío 2",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/carton/frio/iq_tx_2.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/carton/frio/iq_rx_2.bin",
    },
    {
        "id": "plástico frío 1",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/plastico/frio/iq_tx_1.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/plastico/frio/iq_rx_1.bin",
    },
    {
        "id": "plástico frío 2",
        "tx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/plastico/frio/iq_tx_2.bin",
        "rx_path": "/home/hugo/Escritorio/ISAC/Codigo-ISAC/data/plastico/frio/iq_rx_2.bin",
    },
]


# =============================================================================
# Función auxiliar: filtrar solo características escalares
# =============================================================================

def keep_scalar_features(features: dict) -> dict:
    """
    Elimina arrays o matrices auxiliares y conserva solo características escalares.
    """

    scalar_features = {}

    for key, value in features.items():
        if isinstance(value, np.ndarray):
            continue

        if np.isscalar(value):
            scalar_features[key] = float(value)

    return scalar_features


# =============================================================================
# Cálculo de características para todos los pares
# =============================================================================

rows = []

for pair in signal_pairs:
    measure_id = pair["id"]
    tx_path = pair["tx_path"]
    rx_path = pair["rx_path"]

    print("=" * 80)
    print(f"Procesando {measure_id}")
    print(f"TX: {tx_path}")
    print(f"RX: {rx_path}")

    try:
        # ------------------------------------------------------------
        # 1. Obtener matriz de canal H
        # ------------------------------------------------------------
        H = compute_channel_matrix_from_iq_paths(
            tx_path,
            rx_path,
            yaml_path,
            n_symbols=100,
            storage_format_tx="auto",
            storage_format_rx="auto",
            fftshift=False,
            normalize_fft=False,
            output_order="mk",   # H[m, k]
            return_aux=False,
            verbose=False,
        )

        # ------------------------------------------------------------
        # 2. Extraer características generales de H
        # ------------------------------------------------------------
        features = extract_channel_matrix_features(
            H,
            input_order="mk",
            return_phase_arrays=False,
        )

        features = keep_scalar_features(features)

        # ------------------------------------------------------------
        # 3. Añadir metadatos de la medida
        # ------------------------------------------------------------
        row = {
            "id": measure_id,
            "tx_path": tx_path,
            "rx_path": rx_path,
            "status": "ok",
        }

        row.update(features)

    except Exception as e:
        row = {
            "id": measure_id,
            "tx_path": tx_path,
            "rx_path": rx_path,
            "status": "error",
            "error_message": str(e),
        }

        print(f"Error en {measure_id}: {e}")

    rows.append(row)


# =============================================================================
# Crear tabla y exportar a Excel
# =============================================================================

df = pd.DataFrame(rows)

# Reordenar columnas: metadatos primero
metadata_cols = ["id", "tx_path", "rx_path", "status"]

if "error_message" in df.columns:
    metadata_cols.append("error_message")

feature_cols = [col for col in df.columns if col not in metadata_cols]

df = df[metadata_cols + feature_cols]

# Guardar Excel
with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="caracteristicas_H", index=False)

    # Ajuste sencillo de anchura de columnas
    worksheet = writer.sheets["caracteristicas_H"]

    for column_cells in worksheet.columns:
        max_length = 0
        column_letter = column_cells[0].column_letter

        for cell in column_cells:
            value = cell.value
            if value is not None:
                max_length = max(max_length, len(str(value)))

        adjusted_width = min(max_length + 2, 40)
        worksheet.column_dimensions[column_letter].width = adjusted_width

print("=" * 80)
print("Excel generado correctamente")
print(f"Ruta: {Path(output_excel_path).resolve()}")
print(f"Shape de la tabla: {df.shape}")