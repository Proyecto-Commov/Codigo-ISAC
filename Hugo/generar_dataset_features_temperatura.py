from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


# =============================================================================
# Imports locales de la carpeta Hugo
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from channel_estimator import compute_channel_matrix_from_iq_paths
from channel_features_complete import extract_channel_matrix_features


IQ_STORAGE_CHOICES = ["auto", "complex64", "sc16_interleaved", "sc8_interleaved"]
SplitMode = Literal["stratified", "random"]


# =============================================================================
# Rutas por defecto
# =============================================================================

def default_paths() -> tuple[Path, Path, Path]:
    """
    Supone esta estructura:

        carpeta_raiz/
        ├── BBDD/
        └── Codigo-ISAC/
            ├── data/
            │   └── Modulator.yaml
            └── Hugo/
                └── generar_dataset_features_temperatura.py
    """
    hugo_dir = THIS_DIR
    codigo_isac_dir = hugo_dir.parent
    root_dir = codigo_isac_dir.parent

    bbdd_dir = root_dir / "BBDD"
    yaml_path = codigo_isac_dir / "data" / "Modulator.yaml"
    output_dir = hugo_dir / "dataset_features_temperatura"

    return bbdd_dir, yaml_path, output_dir


def _safe_relative_path(path: Path, base: Path) -> str:
    """Devuelve path relativo a base si es posible; si no, devuelve path absoluto."""
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


# =============================================================================
# Utilidades de base de datos
# =============================================================================

def parse_temperature_from_folder(folder_name: str) -> float:
    """
    Extrae la temperatura a partir del nombre de la carpeta.

    Acepta nombres como:
        "25"
        "25.5"
        "25,5"
        "temp_25.5C"
    """
    text = folder_name.strip().replace(",", ".")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)

    if match is None:
        raise ValueError(f"No se ha podido extraer temperatura de: {folder_name}")

    return float(match.group(0))


def get_iq_pairs(temp_dir: Path) -> list[tuple[str, Path, Path]]:
    """
    Empareja archivos iq_tx_N.bin e iq_rx_N.bin dentro de una carpeta.

    Devuelve:
        [(sample_id, tx_path, rx_path), ...]
    """
    tx_files: dict[str, Path] = {}
    rx_files: dict[str, Path] = {}

    for path in temp_dir.iterdir():
        if not path.is_file():
            continue

        name = path.name.strip()

        m_tx = re.match(r"^iq_tx_(\d+)\.bin$", name, flags=re.IGNORECASE)
        if m_tx:
            tx_files[m_tx.group(1)] = path
            continue

        m_rx = re.match(r"^iq_rx_(\d+)\.bin$", name, flags=re.IGNORECASE)
        if m_rx:
            rx_files[m_rx.group(1)] = path
            continue

    common_ids = sorted(set(tx_files) & set(rx_files), key=lambda x: int(x))

    return [(sample_id, tx_files[sample_id], rx_files[sample_id]) for sample_id in common_ids]


def discover_samples(
    *,
    bbdd_dir: Path,
    max_samples_per_temperature: int | None,
) -> pd.DataFrame:
    """
    Recorre BBDD y genera una tabla de metadatos antes de calcular características.
    """
    if not bbdd_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta BBDD: {bbdd_dir}")

    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    temp_dirs = [p for p in bbdd_dir.iterdir() if p.is_dir()]

    def _temp_sort_key(path: Path) -> tuple[int, float, str]:
        try:
            return (0, parse_temperature_from_folder(path.name), path.name)
        except Exception:
            return (1, float("inf"), path.name)

    temp_dirs = sorted(temp_dirs, key=_temp_sort_key)

    for temp_dir in temp_dirs:
        try:
            temperature = parse_temperature_from_folder(temp_dir.name)
        except Exception as exc:
            errors.append(f"Carpeta ignorada: {temp_dir} | {exc}")
            continue

        pairs = get_iq_pairs(temp_dir)

        if max_samples_per_temperature is not None:
            pairs = pairs[:max_samples_per_temperature]

        for sample_id, tx_path, rx_path in pairs:
            sample_uid = f"{temp_dir.name}__{sample_id}"
            rows.append(
                {
                    "sample_uid": sample_uid,
                    "temperature": temperature,
                    "temperature_folder": temp_dir.name,
                    "sample_id": sample_id,
                    "tx_path": str(tx_path.resolve()),
                    "rx_path": str(rx_path.resolve()),
                    "tx_path_rel_to_bbdd_parent": _safe_relative_path(tx_path, bbdd_dir.parent),
                    "rx_path_rel_to_bbdd_parent": _safe_relative_path(rx_path, bbdd_dir.parent),
                }
            )

    if errors:
        print("AVISOS durante el descubrimiento de muestras:")
        for msg in errors:
            print(f"  {msg}")
        print()

    return pd.DataFrame(rows)


# =============================================================================
# Split train / validation / test
# =============================================================================

def _random_split_indices(
    n: int,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split aleatorio con tamaños aproximados a train/val/test."""
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)

    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    if n == 1:
        return idx, np.array([], dtype=int), np.array([], dtype=int)

    if n == 2:
        return idx[:1], np.array([], dtype=int), idx[1:]

    n_val = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
    n_test = max(1, int(round(test_frac * n))) if test_frac > 0 else 0
    n_train = n - n_val - n_test

    while n_train < 1 and (n_val > 1 or n_test > 1):
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        n_train = n - n_val - n_test

    if n_train < 1:
        n_train = 1
        if n_val > 0:
            n_val -= 1
        else:
            n_test -= 1

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    return train_idx, val_idx, test_idx


def assign_splits(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
    split_mode: SplitMode = "stratified",
) -> pd.DataFrame:
    """
    Añade la columna `split` con valores: train, val, test.

    Modo por defecto: stratified.
        Intenta conservar la proporción de carpetas de temperatura en train/val/test.
        Si no es posible, cae automáticamente a split aleatorio.

    Nota: con pocas muestras por temperatura, los porcentajes pueden no ser exactos.
    """
    if df.empty:
        raise ValueError("No hay muestras para dividir en train/val/test.")

    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Las fracciones deben sumar 1. Recibido: "
            f"train={train_frac}, val={val_frac}, test={test_frac}, suma={total}"
        )

    df = df.copy()
    n = len(df)

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    if split_mode == "stratified" and n >= 3:
        try:
            from sklearn.model_selection import train_test_split

            indices = np.arange(n)
            labels = df["temperature_folder"].astype(str).to_numpy()

            # La estratificación exige al menos 2 muestras por grupo para el primer corte.
            value_counts = pd.Series(labels).value_counts()
            can_stratify_first = (len(value_counts) > 1) and (value_counts.min() >= 2)

            stratify_first = labels if can_stratify_first else None

            train_idx, tmp_idx = train_test_split(
                indices,
                train_size=train_frac,
                random_state=random_state,
                shuffle=True,
                stratify=stratify_first,
            )

            if len(tmp_idx) == 0:
                val_idx = np.array([], dtype=int)
                test_idx = np.array([], dtype=int)
            else:
                val_ratio_within_tmp = val_frac / (val_frac + test_frac)
                tmp_labels = labels[tmp_idx]
                tmp_value_counts = pd.Series(tmp_labels).value_counts()
                can_stratify_second = (len(tmp_value_counts) > 1) and (tmp_value_counts.min() >= 2)
                stratify_second = tmp_labels if can_stratify_second else None

                if len(tmp_idx) == 1:
                    val_idx = tmp_idx
                    test_idx = np.array([], dtype=int)
                else:
                    val_idx, test_idx = train_test_split(
                        tmp_idx,
                        train_size=val_ratio_within_tmp,
                        random_state=random_state + 1,
                        shuffle=True,
                        stratify=stratify_second,
                    )

        except Exception as exc:
            print(f"AVISO: no se pudo hacer split estratificado. Se usa split aleatorio. Motivo: {exc}")
            train_idx, val_idx, test_idx = _random_split_indices(
                n,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                random_state=random_state,
            )
    elif split_mode == "random":
        train_idx, val_idx, test_idx = _random_split_indices(
            n,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            random_state=random_state,
        )
    else:
        raise ValueError("split_mode debe ser 'stratified' o 'random'.")

    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # Seguridad por si quedase alguna fila sin asignar.
    if (df["split"] == "").any():
        df.loc[df["split"] == "", "split"] = "train"

    return df


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de conteos por split y temperatura."""
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(["split", "temperature_folder", "temperature"], dropna=False)
        .size()
        .reset_index(name="n_samples")
        .sort_values(["split", "temperature"])
    )
    return summary


# =============================================================================
# Extracción de características
# =============================================================================

def extract_features_for_dataset(
    metadata_df: pd.DataFrame,
    *,
    yaml_path: Path,
    n_symbols: int | None,
    start_sample_tx: int,
    start_sample_rx: int,
    storage_format_tx: str,
    storage_format_rx: str,
    verbose_pipeline: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula H y características para cada muestra de metadata_df.

    Devuelve:
        dataset_df: filas válidas con metadatos + características
        errors_df: filas que fallaron con mensaje de error
    """
    if metadata_df.empty:
        raise ValueError("metadata_df está vacío. No hay pares TX/RX que procesar.")

    if not yaml_path.exists():
        raise FileNotFoundError(f"No existe el YAML: {yaml_path}")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    total = len(metadata_df)

    for i, meta in metadata_df.reset_index(drop=True).iterrows():
        sample_uid = str(meta["sample_uid"])
        tx_path = Path(str(meta["tx_path"]))
        rx_path = Path(str(meta["rx_path"]))

        try:
            H = compute_channel_matrix_from_iq_paths(
                tx_path,
                rx_path,
                yaml_path,
                n_symbols=n_symbols,
                start_sample_tx=start_sample_tx,
                start_sample_rx=start_sample_rx,
                storage_format_tx=storage_format_tx,
                storage_format_rx=storage_format_rx,
                fftshift=False,
                normalize_fft=False,
                trim_to_complete_frames=True,
                return_aux=False,
                output_order="mk",
                verbose=verbose_pipeline,
            )

            features = extract_channel_matrix_features(H, input_order="mk")

            row = meta.to_dict()
            row.update(features)
            rows.append(row)

            if not verbose_pipeline:
                current = i + 1
                if current % 10 == 0 or current == total:
                    print(f"Procesadas {current}/{total} muestras")

        except Exception as exc:
            err = meta.to_dict()
            err["error"] = repr(exc)
            errors.append(err)
            print(f"ERROR en muestra {sample_uid}: {exc}")

    dataset_df = pd.DataFrame(rows)
    errors_df = pd.DataFrame(errors)

    return dataset_df, errors_df


# =============================================================================
# Guardado de resultados
# =============================================================================

def ordered_columns(df: pd.DataFrame) -> list[str]:
    """Coloca primero los metadatos y después las características."""
    preferred = [
        "sample_uid",
        "split",
        "temperature",
        "temperature_folder",
        "sample_id",
        "tx_path",
        "rx_path",
        "tx_path_rel_to_bbdd_parent",
        "rx_path_rel_to_bbdd_parent",
    ]

    cols = list(df.columns)
    first = [c for c in preferred if c in cols]
    rest = [c for c in cols if c not in first]
    return first + rest


def save_outputs(
    *,
    dataset_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    output_dir: Path,
    csv_name: str,
    excel_name: str,
    write_excel: bool,
    config: dict[str, Any],
) -> dict[str, Path]:
    """Guarda CSV, Excel opcional, resumen y configuración."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = dataset_df[ordered_columns(dataset_df)]

    dataset_csv = output_dir / csv_name
    errors_csv = output_dir / "errores_procesado.csv"
    summary_csv = output_dir / "resumen_splits.csv"
    feature_cols_json = output_dir / "columnas_features.json"
    config_json = output_dir / "config_dataset.json"

    dataset_df.to_csv(dataset_csv, index=False)
    errors_df.to_csv(errors_csv, index=False)

    summary_df = split_summary(dataset_df)
    summary_df.to_csv(summary_csv, index=False)

    metadata_cols = {
        "sample_uid",
        "split",
        "temperature",
        "temperature_folder",
        "sample_id",
        "tx_path",
        "rx_path",
        "tx_path_rel_to_bbdd_parent",
        "rx_path_rel_to_bbdd_parent",
    }
    feature_cols = [c for c in dataset_df.columns if c not in metadata_cols]

    with open(feature_cols_json, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=4, ensure_ascii=False)

    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False, default=str)

    paths = {
        "dataset_csv": dataset_csv,
        "errors_csv": errors_csv,
        "summary_csv": summary_csv,
        "feature_cols_json": feature_cols_json,
        "config_json": config_json,
    }

    if write_excel:
        dataset_xlsx = output_dir / excel_name
        try:
            with pd.ExcelWriter(dataset_xlsx, engine="openpyxl") as writer:
                dataset_df.to_excel(writer, index=False, sheet_name="dataset")
                summary_df.to_excel(writer, index=False, sheet_name="resumen_splits")
                errors_df.to_excel(writer, index=False, sheet_name="errores")

                # Ajuste simple de anchura de columnas.
                for sheet_name, sheet_df in {
                    "dataset": dataset_df,
                    "resumen_splits": summary_df,
                    "errores": errors_df,
                }.items():
                    ws = writer.sheets[sheet_name]
                    for col_idx, col_name in enumerate(sheet_df.columns, start=1):
                        values = sheet_df[col_name].astype(str).head(100).tolist()
                        max_len = max([len(str(col_name))] + [len(v) for v in values])
                        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max(max_len + 2, 10), 50)
                    ws.freeze_panes = "A2"

            paths["dataset_xlsx"] = dataset_xlsx
        except Exception as exc:
            print(f"AVISO: no se pudo guardar Excel .xlsx. Se mantiene el CSV. Motivo: {exc}")

    return paths


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    default_bbdd, default_yaml, default_output = default_paths()

    parser = argparse.ArgumentParser(
        description=(
            "Genera un dataset tabular con rutas TX/RX, temperatura, split "
            "train/val/test y características de channel_features_complete.py."
        )
    )

    parser.add_argument("--bbdd-dir", type=Path, default=default_bbdd)
    parser.add_argument("--yaml-path", type=Path, default=default_yaml)
    parser.add_argument("--output-dir", type=Path, default=default_output)

    parser.add_argument("--n-symbols", type=int, default=None)
    parser.add_argument("--start-sample-tx", type=int, default=0)
    parser.add_argument("--start-sample-rx", type=int, default=0)
    parser.add_argument("--storage-format-tx", type=str, default="auto", choices=IQ_STORAGE_CHOICES)
    parser.add_argument("--storage-format-rx", type=str, default="auto", choices=IQ_STORAGE_CHOICES)

    parser.add_argument("--max-samples-per-temperature", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-mode", type=str, default="stratified", choices=["stratified", "random"])
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)

    parser.add_argument("--csv-name", type=str, default="dataset_features_temperatura.csv")
    parser.add_argument("--excel-name", type=str, default="dataset_features_temperatura.xlsx")
    parser.add_argument("--no-excel", action="store_true", help="No genera archivo Excel; solo CSV.")
    parser.add_argument("--verbose-pipeline", action="store_true")

    args = parser.parse_args()

    print("============================================================")
    print("GENERACIÓN DE DATASET DE FEATURES")
    print("============================================================")
    print(f"BBDD: {args.bbdd_dir}")
    print(f"YAML: {args.yaml_path}")
    print(f"Salida: {args.output_dir}")
    print()

    metadata_df = discover_samples(
        bbdd_dir=args.bbdd_dir,
        max_samples_per_temperature=args.max_samples_per_temperature,
    )

    if metadata_df.empty:
        raise ValueError("No se encontró ningún par iq_tx_N.bin / iq_rx_N.bin válido.")

    print("Muestras encontradas antes de procesar:")
    print(f"  Total: {len(metadata_df)}")
    print(f"  Temperaturas: {metadata_df['temperature_folder'].nunique()}")
    print()

    dataset_df, errors_df = extract_features_for_dataset(
        metadata_df,
        yaml_path=args.yaml_path,
        n_symbols=args.n_symbols,
        start_sample_tx=args.start_sample_tx,
        start_sample_rx=args.start_sample_rx,
        storage_format_tx=args.storage_format_tx,
        storage_format_rx=args.storage_format_rx,
        verbose_pipeline=args.verbose_pipeline,
    )

    if dataset_df.empty:
        raise ValueError("No se pudo procesar ninguna muestra correctamente.")

    dataset_df = assign_splits(
        dataset_df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.random_state,
        split_mode=args.split_mode,
    )

    config = {
        "bbdd_dir": str(args.bbdd_dir),
        "yaml_path": str(args.yaml_path),
        "output_dir": str(args.output_dir),
        "n_symbols": args.n_symbols,
        "start_sample_tx": args.start_sample_tx,
        "start_sample_rx": args.start_sample_rx,
        "storage_format_tx": args.storage_format_tx,
        "storage_format_rx": args.storage_format_rx,
        "max_samples_per_temperature": args.max_samples_per_temperature,
        "random_state": args.random_state,
        "split_mode": args.split_mode,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
    }

    paths = save_outputs(
        dataset_df=dataset_df,
        errors_df=errors_df,
        output_dir=args.output_dir,
        csv_name=args.csv_name,
        excel_name=args.excel_name,
        write_excel=not args.no_excel,
        config=config,
    )

    print()
    print("============================================================")
    print("RESUMEN FINAL")
    print("============================================================")
    print(f"Muestras válidas: {len(dataset_df)}")
    print(f"Muestras con error: {len(errors_df)}")
    print(f"Número de características: {len(json.loads(Path(paths['feature_cols_json']).read_text(encoding='utf-8')))}")
    print()
    print("Reparto por split:")
    print(dataset_df["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int).to_string())
    print()
    print("Archivos guardados:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
