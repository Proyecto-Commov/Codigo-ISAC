from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_CSV = THIS_DIR / "dataset_features_temperatura" / "dataset_features_temperatura.csv"
DEFAULT_FEATURES_JSON = THIS_DIR / "dataset_features_temperatura" / "columnas_features.json"
DEFAULT_OUTPUT_DIR = THIS_DIR / "correlaciones_features_temperatura"

DEFAULT_METADATA_COLS = {
    "sample_uid",
    "sample_id",
    "tx_path",
    "rx_path",
    "tx_path_rel_to_bbdd_parent",
    "rx_path_rel_to_bbdd_parent",
    "split",
    "temperature_folder",
    "N_subcarriers",
    "M_symbols",
    "num_valid",
    "valid_ratio",
}


def load_feature_columns(features_json: Path | None) -> list[str] | None:
    if features_json is None or not Path(features_json).exists():
        return None

    with open(features_json, "r", encoding="utf-8") as f:
        cols = json.load(f)

    if not isinstance(cols, list):
        raise ValueError(f"{features_json} no contiene una lista JSON válida.")

    return [str(c) for c in cols]


def filter_by_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    split = split.lower().strip()

    if split == "all":
        return df.copy()

    if "split" not in df.columns:
        raise ValueError("El CSV no tiene columna 'split'.")

    out = df[df["split"].astype(str).str.lower() == split].copy()

    if out.empty:
        raise ValueError(f"No quedan filas tras aplicar split='{split}'.")

    return out


def infer_variable_columns(
    df: pd.DataFrame,
    *,
    feature_cols_from_json: list[str] | None,
    include_temperature: bool,
    exclude_cols: Iterable[str],
) -> list[str]:
    """
    Selecciona columnas para la matriz de correlación.

    Prioridad:
      1. Si existe columnas_features.json, usa esas columnas.
      2. Si no existe, usa columnas convertibles a numéricas excluyendo metadatos.
    """
    exclude = set(exclude_cols)

    if feature_cols_from_json is not None:
        feature_cols = [c for c in feature_cols_from_json if c in df.columns]
        missing = [c for c in feature_cols_from_json if c not in df.columns]
        if missing:
            print(f"AVISO: {len(missing)} columnas de columnas_features.json no están en el CSV.")
    else:
        feature_cols = []
        for c in df.columns:
            if c in exclude or c == "temperature" or str(c).startswith("Unnamed"):
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                feature_cols.append(c)

    cols: list[str] = []

    if include_temperature:
        if "temperature" not in df.columns:
            raise ValueError("El CSV no contiene la columna 'temperature'.")
        cols.append("temperature")

    cols.extend([c for c in feature_cols if c not in cols])

    if not cols:
        raise ValueError("No se ha encontrado ninguna columna numérica para correlacionar.")

    return cols


def make_numeric_dataframe(df: pd.DataFrame, cols: list[str], *, min_non_nan: int) -> pd.DataFrame:
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    keep_cols = []
    removed_constant = []
    removed_empty = []

    for c in X.columns:
        s = X[c]
        if int(s.notna().sum()) < min_non_nan:
            removed_empty.append(c)
            continue
        if s.dropna().nunique() <= 1:
            removed_constant.append(c)
            continue
        keep_cols.append(c)

    if removed_empty:
        print(f"Columnas eliminadas por pocos datos válidos: {len(removed_empty)}")
    if removed_constant:
        print(f"Columnas eliminadas por ser constantes: {len(removed_constant)}")

    X = X[keep_cols]

    if X.shape[1] < 2:
        raise ValueError("Quedan menos de dos variables válidas. No se puede calcular Pearson.")

    return X


def correlation_pairs(corr: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    if len(variables) < 2:
        return pd.DataFrame(columns=["var1", "var2", "r", "abs_r"])

    corr_sub = corr.loc[variables, variables]
    mask_upper = np.triu(np.ones(corr_sub.shape, dtype=bool), k=1)

    pairs = (
        corr_sub.where(mask_upper)
        .stack(dropna=True)
        .reset_index()
        .rename(columns={"level_0": "var1", "level_1": "var2", 0: "r"})
    )

    if pairs.empty:
        return pd.DataFrame(columns=["var1", "var2", "r", "abs_r"])

    pairs["abs_r"] = pairs["r"].abs()
    return pairs.sort_values("abs_r", ascending=False).reset_index(drop=True)


def correlation_band(r: float) -> str:
    a = abs(float(r))
    if np.isclose(a, 1.0):
        return "r = ±1.00"
    if a >= 0.99:
        return "0.99 <= |r| < 1.00"
    if a >= 0.95:
        return "0.95 <= |r| < 0.99"
    if a >= 0.90:
        return "0.90 <= |r| < 0.95"
    if a >= 0.80:
        return "0.80 <= |r| < 0.90"
    if a >= 0.70:
        return "0.70 <= |r| < 0.80"
    return "|r| < 0.70"


def variable_summary(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        s = X[c]
        rows.append({
            "variable": c,
            "n_valid": int(s.notna().sum()),
            "n_nan": int(s.isna().sum()),
            "mean": float(s.mean(skipna=True)),
            "std": float(s.std(skipna=True)),
            "min": float(s.min(skipna=True)),
            "max": float(s.max(skipna=True)),
        })
    return pd.DataFrame(rows)


def save_heatmap(corr: pd.DataFrame, output_path: Path, *, title: str, max_labels: int = 120) -> None:
    import matplotlib.pyplot as plt

    n = corr.shape[0]
    fig_size = max(8, min(28, 0.28 * n + 6))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Pearson r")

    ax.set_title(title)

    if n <= max_labels:
        ticks = np.arange(n)
    else:
        step = int(np.ceil(n / max_labels))
        ticks = np.arange(0, n, step)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([corr.columns[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([corr.index[i] for i in ticks], fontsize=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_temperature_barplot(corr_temp: pd.DataFrame, output_path: Path, *, top_n: int) -> None:
    import matplotlib.pyplot as plt

    if corr_temp.empty:
        return

    data = corr_temp.head(top_n).iloc[::-1]
    fig_height = max(6, 0.32 * len(data))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    ax.barh(data["feature"], data["r"])
    ax.axvline(0, linewidth=0.8)
    ax.axvline(0.5, linewidth=0.6, linestyle="--", alpha=0.6)
    ax.axvline(-0.5, linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel("Correlación de Pearson con temperature")
    ax.set_title(f"Top {len(data)} correlaciones con temperature")
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calcula correlaciones de Pearson cruzadas a partir de dataset_features_temperatura.csv."
    )

    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--features-json", type=Path, default=DEFAULT_FEATURES_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--min-non-nan", type=int, default=3)
    parser.add_argument("--no-temperature", action="store_true")
    parser.add_argument("--no-excel", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--top-n-temperature", type=int, default=30)

    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"No existe el CSV de entrada: {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = filter_by_split(df, args.split)

    feature_cols_from_json = load_feature_columns(args.features_json)

    variable_cols = infer_variable_columns(
        df,
        feature_cols_from_json=feature_cols_from_json,
        include_temperature=not args.no_temperature,
        exclude_cols=DEFAULT_METADATA_COLS,
    )

    X = make_numeric_dataframe(df, variable_cols, min_non_nan=args.min_non_nan)

    corr = X.corr(method="pearson", min_periods=args.min_non_nan)

    all_pairs = correlation_pairs(corr, list(corr.columns))
    all_pairs["banda"] = all_pairs["r"].apply(correlation_band)
    high_pairs = all_pairs[all_pairs["abs_r"] >= args.threshold].copy()

    feature_variables = [c for c in corr.columns if c != "temperature"]
    feature_pairs = correlation_pairs(corr, feature_variables)
    if not feature_pairs.empty:
        feature_pairs["banda"] = feature_pairs["r"].apply(correlation_band)
    high_feature_pairs = feature_pairs[feature_pairs["abs_r"] >= args.threshold].copy()

    if "temperature" in corr.columns:
        corr_temp = (
            corr["temperature"]
            .drop(labels=["temperature"], errors="ignore")
            .dropna()
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .reset_index()
            .rename(columns={"index": "feature", "temperature": "r"})
        )
        corr_temp["abs_r"] = corr_temp["r"].abs()
        corr_temp["banda"] = corr_temp["r"].apply(correlation_band)
    else:
        corr_temp = pd.DataFrame(columns=["feature", "r", "abs_r", "banda"])

    summary = variable_summary(X)
    suffix = args.split.lower()
    threshold_label = str(args.threshold).replace(".", "p")

    corr_csv = args.output_dir / f"matriz_correlaciones_pearson_{suffix}.csv"
    all_pairs_csv = args.output_dir / f"pares_correlaciones_todas_{suffix}.csv"
    high_pairs_csv = args.output_dir / f"pares_correlaciones_abs_ge_{threshold_label}_{suffix}.csv"
    high_feature_pairs_csv = args.output_dir / f"pares_features_abs_ge_{threshold_label}_{suffix}.csv"
    corr_temp_csv = args.output_dir / f"correlacion_con_temperature_{suffix}.csv"
    summary_csv = args.output_dir / f"resumen_variables_{suffix}.csv"

    corr.to_csv(corr_csv, index=True)
    all_pairs.to_csv(all_pairs_csv, index=False)
    high_pairs.to_csv(high_pairs_csv, index=False)
    high_feature_pairs.to_csv(high_feature_pairs_csv, index=False)
    corr_temp.to_csv(corr_temp_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    if not args.no_excel:
        excel_path = args.output_dir / f"correlaciones_pearson_{suffix}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            corr.to_excel(writer, sheet_name="matriz_pearson")
            corr_temp.to_excel(writer, sheet_name="corr_temperature", index=False)
            high_feature_pairs.to_excel(writer, sheet_name="features_abs_r_alto", index=False)
            high_pairs.to_excel(writer, sheet_name="pares_abs_r_alto", index=False)
            all_pairs.to_excel(writer, sheet_name="todos_los_pares", index=False)
            summary.to_excel(writer, sheet_name="resumen_variables", index=False)
        print(f"Excel guardado en: {excel_path}")

    if args.plots:
        heatmap_path = args.output_dir / f"heatmap_correlaciones_pearson_{suffix}.png"
        save_heatmap(corr, heatmap_path, title=f"Matriz de correlaciones de Pearson ({suffix})")
        print(f"Heatmap guardado en: {heatmap_path}")

        if not corr_temp.empty:
            temp_plot_path = args.output_dir / f"correlaciones_con_temperature_{suffix}.png"
            save_temperature_barplot(corr_temp, temp_plot_path, top_n=args.top_n_temperature)
            print(f"Figura correlación-temperature guardada en: {temp_plot_path}")

    print()
    print("============================================================")
    print("CORRELACIONES DE PEARSON")
    print("============================================================")
    print(f"CSV entrada: {args.input_csv}")
    print(f"Split usado: {args.split}")
    print(f"Filas usadas: {len(df)}")
    print(f"Variables usadas: {len(corr.columns)}")
    print(f"Matriz guardada en: {corr_csv}")
    print(f"Pares |r| >= {args.threshold:g}: {len(high_pairs)}")
    print(f"Pares entre features |r| >= {args.threshold:g}: {len(high_feature_pairs)}")
    print()

    if not corr_temp.empty:
        print(f"Top {min(args.top_n_temperature, len(corr_temp))} correlaciones con temperature:")
        print(corr_temp.head(args.top_n_temperature).to_string(index=False))
        print()

    if not high_feature_pairs.empty:
        print(f"Primeros pares entre features con |r| >= {args.threshold:g}:")
        print(high_feature_pairs.head(30).to_string(index=False))
    else:
        print(f"No se han encontrado pares entre features con |r| >= {args.threshold:g}.")


if __name__ == "__main__":
    main()
