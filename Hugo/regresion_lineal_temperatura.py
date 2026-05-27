from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Imports locales de la carpeta Hugo
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from channel_estimator import compute_channel_matrix_from_iq_paths
from channel_features_complete import extract_channel_matrix_features


# =============================================================================
# Utilidades de rutas y base de datos
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
                └── regresion_lineal_temperatura.py
    """
    hugo_dir = THIS_DIR
    codigo_isac_dir = hugo_dir.parent
    root_dir = codigo_isac_dir.parent

    bbdd_dir = root_dir / "BBDD"
    yaml_path = codigo_isac_dir / "data" / "Modulator.yaml"
    output_dir = hugo_dir / "resultados_regresion_lineal"

    return bbdd_dir, yaml_path, output_dir


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
    Devuelve una lista de tuplas:

        (id_muestra, tx_path, rx_path)
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


def collect_feature_table(
    *,
    bbdd_dir: Path,
    yaml_path: Path,
    n_symbols: int | None,
    start_sample_tx: int,
    start_sample_rx: int,
    storage_format_tx: str,
    storage_format_rx: str,
    max_samples_per_temperature: int | None,
    verbose_pipeline: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recorre BBDD, calcula H para cada par TX/RX, extrae características y
    devuelve:
        - df: tabla con características + metadatos
        - errors_df: errores encontrados durante el recorrido
    """
    if not bbdd_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta BBDD: {bbdd_dir}")

    if not yaml_path.exists():
        raise FileNotFoundError(f"No existe el YAML: {yaml_path}")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    temp_dirs = [p for p in bbdd_dir.iterdir() if p.is_dir()]
    temp_dirs = sorted(temp_dirs, key=lambda p: parse_temperature_from_folder(p.name))

    print(f"Carpeta BBDD: {bbdd_dir}")
    print(f"YAML: {yaml_path}")
    print(f"Carpetas de temperatura encontradas: {len(temp_dirs)}")
    print()

    for temp_dir in temp_dirs:
        try:
            temperature = parse_temperature_from_folder(temp_dir.name)
        except Exception as exc:
            errors.append({
                "temperature_folder": temp_dir.name,
                "sample_id": "",
                "tx_path": "",
                "rx_path": "",
                "error": repr(exc),
            })
            continue

        pairs = get_iq_pairs(temp_dir)

        if max_samples_per_temperature is not None:
            pairs = pairs[:max_samples_per_temperature]

        print(f"Temperatura {temperature:g} | carpeta {temp_dir.name} | pares: {len(pairs)}")

        for i, (sample_id, tx_path, rx_path) in enumerate(pairs, start=1):
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

                features = extract_channel_matrix_features(
                    H,
                    input_order="mk",
                )

                row: dict[str, Any] = {
                    "temperature": temperature,
                    "temperature_folder": temp_dir.name,
                    "sample_id": sample_id,
                    "tx_path": str(tx_path),
                    "rx_path": str(rx_path),
                }
                row.update(features)
                rows.append(row)

                if not verbose_pipeline and (i % 10 == 0 or i == len(pairs)):
                    print(f"  Procesados {i}/{len(pairs)}")

            except Exception as exc:
                errors.append({
                    "temperature_folder": temp_dir.name,
                    "sample_id": sample_id,
                    "tx_path": str(tx_path),
                    "rx_path": str(rx_path),
                    "error": repr(exc),
                })
                print(f"  ERROR muestra {sample_id}: {exc}")

    df = pd.DataFrame(rows)
    errors_df = pd.DataFrame(errors)

    return df, errors_df


# =============================================================================
# Regresión lineal multivariable
# =============================================================================

def build_regression_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Separa X e y, dejando como variables de entrada únicamente las características.
    """
    if df.empty:
        raise ValueError("No hay muestras válidas para entrenar.")

    metadata_cols = {
        "temperature",
        "temperature_folder",
        "sample_id",
        "tx_path",
        "rx_path",
    }

    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Elimina columnas completamente vacías.
    non_empty_cols = [c for c in X.columns if not X[c].isna().all()]
    X = X[non_empty_cols]

    if X.shape[1] == 0:
        raise ValueError("No queda ninguna característica numérica válida.")

    y = pd.to_numeric(df["temperature"], errors="raise")

    return X, y, non_empty_cols


def fit_linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
    split_by_temperature: bool,
    groups: pd.Series | None,
) -> dict[str, Any]:
    """
    Entrena una regresión lineal con:
        imputación de NaN por mediana + estandarización + LinearRegression.
    """
    n_samples = len(X)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    use_test = n_samples >= 5 and test_size > 0.0

    if use_test:
        if split_by_temperature:
            if groups is None:
                raise ValueError("split_by_temperature=True requiere grupos.")

            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state,
            )
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
            )

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            "n_samples_total": int(n_samples),
            "n_samples_train": int(len(X_train)),
            "n_samples_test": int(len(X_test)),
            "r2_train": float(r2_score(y_train, y_pred_train)) if len(y_train) >= 2 else float("nan"),
            "r2_test": float(r2_score(y_test, y_pred_test)) if len(y_test) >= 2 else float("nan"),
            "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
            "mae_test": float(mean_absolute_error(y_test, y_pred_test)),
            "rmse_train": float(mean_squared_error(y_train, y_pred_train, squared=False)),
            "rmse_test": float(mean_squared_error(y_test, y_pred_test, squared=False)),
        }

        predictions_df = pd.DataFrame(
            {
                "split": ["train"] * len(y_train) + ["test"] * len(y_test),
                "y_true": np.concatenate([np.asarray(y_train), np.asarray(y_test)]),
                "y_pred": np.concatenate([y_pred_train, y_pred_test]),
            }
        )

    else:
        model.fit(X, y)
        y_pred = model.predict(X)

        metrics = {
            "n_samples_total": int(n_samples),
            "n_samples_train": int(n_samples),
            "n_samples_test": 0,
            "r2_train": float(r2_score(y, y_pred)) if len(y) >= 2 else float("nan"),
            "r2_test": float("nan"),
            "mae_train": float(mean_absolute_error(y, y_pred)),
            "mae_test": float("nan"),
            "rmse_train": float(mean_squared_error(y, y_pred, squared=False)),
            "rmse_test": float("nan"),
        }

        predictions_df = pd.DataFrame(
            {
                "split": ["train"] * len(y),
                "y_true": np.asarray(y),
                "y_pred": y_pred,
            }
        )

    return {
        "model": model,
        "metrics": metrics,
        "predictions_df": predictions_df,
    }


def coefficients_table(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """
    Devuelve coeficientes en escala estandarizada y en escala original.

    El modelo entrenado es:
        y = intercept_scaled + sum beta_scaled_j * z_j
        z_j = (x_j - mean_j) / scale_j

    Por tanto, en variables originales:
        beta_original_j = beta_scaled_j / scale_j
    """
    scaler: StandardScaler = model.named_steps["scaler"]
    regressor: LinearRegression = model.named_steps["regressor"]

    beta_scaled = np.asarray(regressor.coef_, dtype=float)
    beta_original = beta_scaled / scaler.scale_

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coef_standardized": beta_scaled,
            "coef_original_units": beta_original,
            "abs_coef_standardized": np.abs(beta_scaled),
        }
    )

    table = table.sort_values("abs_coef_standardized", ascending=False).reset_index(drop=True)

    return table


def original_units_intercept(model: Pipeline) -> float:
    scaler: StandardScaler = model.named_steps["scaler"]
    regressor: LinearRegression = model.named_steps["regressor"]

    beta_scaled = np.asarray(regressor.coef_, dtype=float)
    intercept_original = float(regressor.intercept_ - np.sum(beta_scaled * scaler.mean_ / scaler.scale_))

    return intercept_original


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    default_bbdd, default_yaml, default_output = default_paths()

    parser = argparse.ArgumentParser(
        description=(
            "Regresión lineal multivariable de temperatura a partir de pares "
            "iq_tx_N.bin / iq_rx_N.bin y características de H."
        )
    )

    parser.add_argument("--bbdd-dir", type=Path, default=default_bbdd)
    parser.add_argument("--yaml-path", type=Path, default=default_yaml)
    parser.add_argument("--output-dir", type=Path, default=default_output)

    parser.add_argument("--n-symbols", type=int, default=None)
    parser.add_argument("--start-sample-tx", type=int, default=0)
    parser.add_argument("--start-sample-rx", type=int, default=0)
    parser.add_argument("--storage-format-tx", type=str, default="auto", choices=["auto", "complex64", "sc16_interleaved", "sc8_interleaved"])
    parser.add_argument("--storage-format-rx", type=str, default="auto", choices=["auto", "complex64", "sc16_interleaved", "sc8_interleaved"])

    parser.add_argument("--max-samples-per-temperature", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split-by-temperature", action="store_true", help="Hace el test dejando fuera carpetas completas de temperatura.")
    parser.add_argument("--verbose-pipeline", action="store_true")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, errors_df = collect_feature_table(
        bbdd_dir=args.bbdd_dir,
        yaml_path=args.yaml_path,
        n_symbols=args.n_symbols,
        start_sample_tx=args.start_sample_tx,
        start_sample_rx=args.start_sample_rx,
        storage_format_tx=args.storage_format_tx,
        storage_format_rx=args.storage_format_rx,
        max_samples_per_temperature=args.max_samples_per_temperature,
        verbose_pipeline=args.verbose_pipeline,
    )

    features_csv = args.output_dir / "features_temperatura.csv"
    errors_csv = args.output_dir / "errores_procesado.csv"

    df.to_csv(features_csv, index=False)
    errors_df.to_csv(errors_csv, index=False)

    print()
    print("============================================================")
    print("RESUMEN DE EXTRACCIÓN")
    print("============================================================")
    print(f"Muestras válidas: {len(df)}")
    print(f"Muestras con error: {len(errors_df)}")
    print(f"CSV de características: {features_csv}")
    print(f"CSV de errores: {errors_csv}")

    X, y, feature_cols = build_regression_dataset(df)

    result = fit_linear_regression(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        split_by_temperature=args.split_by_temperature,
        groups=df["temperature_folder"] if args.split_by_temperature else None,
    )

    model: Pipeline = result["model"]
    metrics: dict[str, Any] = result["metrics"]
    predictions_df: pd.DataFrame = result["predictions_df"]

    coef_df = coefficients_table(model, feature_cols)
    intercept_orig = original_units_intercept(model)

    metrics_path = args.output_dir / "metricas_regresion.json"
    coef_path = args.output_dir / "coeficientes_regresion.csv"
    predictions_path = args.output_dir / "predicciones_regresion.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    coef_df.to_csv(coef_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print()
    print("============================================================")
    print("RESULTADOS DE LA REGRESIÓN LINEAL MULTIVARIABLE")
    print("============================================================")
    print(f"Número de muestras total: {metrics['n_samples_total']}")
    print(f"Número de características: {len(feature_cols)}")
    print(f"Muestras train: {metrics['n_samples_train']}")
    print(f"Muestras test: {metrics['n_samples_test']}")
    print()
    print(f"R² train: {metrics['r2_train']:.6f}")
    print(f"R² test : {metrics['r2_test']:.6f}")
    print(f"MAE train  [ºC]: {metrics['mae_train']:.6f}")
    print(f"MAE test   [ºC]: {metrics['mae_test']:.6f}")
    print(f"RMSE train [ºC]: {metrics['rmse_train']:.6f}")
    print(f"RMSE test  [ºC]: {metrics['rmse_test']:.6f}")
    print()
    print("Modelo en variables originales:")
    print(f"  intercept = {intercept_orig:.12g}")
    print("  temperatura ≈ intercept + Σ coef_j · feature_j")
    print()

    print("Coeficientes ordenados por importancia lineal absoluta estandarizada:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 160):
        print(coef_df[["feature", "coef_standardized", "coef_original_units"]].to_string(index=False))

    print()
    print("Archivos guardados:")
    print(f"  {metrics_path}")
    print(f"  {coef_path}")
    print(f"  {predictions_path}")

    if len(df) <= len(feature_cols):
        print()
        print("AVISO: hay menos muestras que características o un número muy parecido.")
        print("La regresión lineal puede ajustar muy bien en train pero tener coeficientes inestables.")


if __name__ == "__main__":
    main()
