from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt


# =============================================================================
# Configuracion por defecto
# =============================================================================

# SELECTED_FEATURES = [
#     "mag_rms",
#     "mag_var",
#     "mag_dynamic_range_db",
#     "mag_entropy_normalized",
#     "phase_coherence",
#     "pdp_los_ratio",
#     "dH_dt_max",
#     "doppler_variance_energy",
#     "doppler_spread",
# ]

SELECTED_FEATURES = [
    "mag_mean",
    "mag_var",
    "mag_rms",
    "mag_energy_total",
    "mag_max",
    "mag_min",
    "mag_percentile_1",
    "mag_percentile_5",
    "mag_percentile_25",
    "mag_percentile_50",
    "mag_percentile_75",
    "mag_percentile_95",
    "mag_percentile_99",
    "mag_dynamic_range_linear",
    "mag_dynamic_range_db",
    "mag_entropy",
    "mag_entropy_normalized",
    "mag_kurtosis",
    "mag_excess_kurtosis",
    "mag_skewness",

    "phase_circular_mean",
    "phase_circular_variance",
    "phase_coherence",
    "phase_unwrap_m_mean",
    "phase_unwrap_m_var",
    "phase_unwrap_m_slope_mean",
    "phase_unwrap_m_slope_var",
    "phase_unwrap_m_slope_rms",
    "phase_unwrap_k_mean",
    "phase_unwrap_k_var",
    "phase_unwrap_k_slope_mean",
    "phase_unwrap_k_slope_var",
    "phase_unwrap_k_slope_rms",

    "phase_jump_m_mean_abs",
    "phase_jump_m_var_abs",
    "phase_jump_m_max_abs",
    "phase_jump_m_rms_abs",
    "phase_jump_k_mean_abs",
    "phase_jump_k_var_abs",
    "phase_jump_k_max_abs",
    "phase_jump_k_rms_abs",

    "pdp_los_peak_power",
    "pdp_los_peak_idx",
    "pdp_total_energy",
    "pdp_papr_linear",
    "pdp_mean_delay",
    "pdp_rms_delay_spread",
    "pdp_los_ratio",

    "svd_sigma_1",
    "svd_sigma_2",
    "svd_sigma_ratio",
    "svd_energy_ratio_mode1",
    "svd_spectral_flatness",

    "dH_dt_mean",
    "dH_dt_max",
    "doppler_variance_energy",
    "doppler_centroid",
    "doppler_spread",
]

META_COLS = [
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

THIS_DIR = Path(__file__).resolve().parent


def default_paths() -> tuple[Path, Path]:
    csv_path = THIS_DIR / "dataset_features_temperatura" / "dataset_features_temperatura.csv"
    output_dir = THIS_DIR / "resultados_ridge_polinomica_grado2"
    return csv_path, output_dir


# =============================================================================
# Utilidades generales
# =============================================================================


def parse_feature_list(text: str | None) -> list[str]:
    if text is None or not text.strip():
        return SELECTED_FEATURES
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_alphas(args: argparse.Namespace) -> np.ndarray:
    if args.alphas is not None and args.alphas.strip():
        vals = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
        if len(vals) == 0:
            raise ValueError("--alphas no contiene ningun valor valido.")
        return np.asarray(vals, dtype=np.float64)

    return np.logspace(
        float(args.alpha_log_min),
        float(args.alpha_log_max),
        int(args.n_alphas),
        dtype=np.float64,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    if len(y_true) == 0:
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan, "n": 0}

    return {
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "n": int(len(y_true)),
    }


# =============================================================================
# Lectura y preparacion de datos
# =============================================================================


def load_dataset(csv_path: Path, feature_cols: list[str]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["split", "temperature", *feature_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias en el CSV:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    df = df.copy()
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["temperature"].notna()].copy()

    valid_splits = {"train", "val", "test"}
    bad_splits = sorted(set(df["split"].dropna()) - valid_splits)
    if bad_splits:
        raise ValueError(f"Hay valores de split no reconocidos: {bad_splits}")

    present_splits = set(df["split"].dropna())
    if "train" not in present_splits:
        raise ValueError("El CSV debe contener muestras con split='train'.")
    if "test" not in present_splits:
        raise ValueError("El CSV debe contener muestras con split='test'.")

    return df


def prepare_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    degree: int,
    interaction_only: bool,
) -> dict[str, Any]:
    """
    Preprocesado correcto para evitar fugas de informacion:

    1) Imputer ajustado solo con train.
    2) Escalado de features originales ajustado solo con train.
    3) Expansion polinomica ajustada solo con train.
    4) Escalado de features polinomicas ajustado solo con train.

    Luego se aplican esas mismas transformaciones a val y test.
    """
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if len(train_df) == 0:
        raise ValueError("No hay muestras train.")
    if len(test_df) == 0:
        raise ValueError("No hay muestras test.")

    imputer = SimpleImputer(strategy="median")
    scaler_raw = StandardScaler()
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False,
        interaction_only=interaction_only,
    )
    scaler_poly = StandardScaler()

    X_train_raw = train_df[feature_cols]
    y_train = train_df["temperature"].to_numpy(dtype=np.float64)

    X_train_imp = imputer.fit_transform(X_train_raw)
    X_train_sc = scaler_raw.fit_transform(X_train_imp)
    X_train_poly = poly.fit_transform(X_train_sc)
    X_train = scaler_poly.fit_transform(X_train_poly).astype(np.float64)

    out: dict[str, Any] = {
        "train": {
            "df": train_df,
            "X": X_train,
            "y": y_train,
        },
        "imputer": imputer,
        "scaler_raw": scaler_raw,
        "poly": poly,
        "scaler_poly": scaler_poly,
        "poly_feature_names": list(poly.get_feature_names_out(feature_cols)),
    }

    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        if len(split_df) == 0:
            out[split_name] = {
                "df": split_df,
                "X": np.empty((0, X_train.shape[1]), dtype=np.float64),
                "y": np.empty((0,), dtype=np.float64),
            }
            continue

        X_imp = imputer.transform(split_df[feature_cols])
        X_sc = scaler_raw.transform(X_imp)
        X_poly = poly.transform(X_sc)
        X = scaler_poly.transform(X_poly).astype(np.float64)
        y = split_df["temperature"].to_numpy(dtype=np.float64)

        out[split_name] = {
            "df": split_df,
            "X": X,
            "y": y,
        }

    return out


# =============================================================================
# Ridge cerrado: CPU NumPy y Torch/CUDA opcional
# =============================================================================


def get_torch_device(prefer_gpu: bool):
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit_ridge_numpy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_select: np.ndarray,
    y_select: np.ndarray,
    *,
    alphas: np.ndarray,
) -> dict[str, Any]:
    Xa = np.column_stack([np.ones(len(X_train)), X_train]).astype(np.float64)
    ya = y_train.reshape(-1, 1).astype(np.float64)

    Xsel_a = np.column_stack([np.ones(len(X_select)), X_select]).astype(np.float64)
    ysel = y_select.reshape(-1, 1).astype(np.float64)

    XtX = Xa.T @ Xa
    Xty = Xa.T @ ya

    reg = np.eye(Xa.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0  # no regularizar intercepto

    best_alpha = None
    best_sol = None
    best_score = np.inf

    for alpha in alphas:
        A = XtX + float(alpha) * reg
        try:
            sol = np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            sol = np.linalg.pinv(A) @ Xty

        pred = Xsel_a @ sol
        score = float(np.sqrt(np.mean((pred - ysel) ** 2)))

        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
            best_sol = sol.squeeze(1)

    if best_sol is None:
        raise RuntimeError("No se pudo ajustar Ridge.")

    return {
        "name": "RidgePoly2_numpy",
        "backend": "numpy_cpu",
        "coef_standardized_poly": best_sol[1:],
        "intercept_standardized_poly": float(best_sol[0]),
        "alpha": float(best_alpha),
        "selection_rmse": float(best_score),
    }


def fit_ridge_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_select: np.ndarray,
    y_select: np.ndarray,
    *,
    alphas: np.ndarray,
    prefer_gpu: bool,
) -> dict[str, Any]:
    import torch

    device = get_torch_device(prefer_gpu)
    dtype = torch.float64

    X = torch.as_tensor(X_train, dtype=dtype, device=device)
    y = torch.as_tensor(y_train.reshape(-1, 1), dtype=dtype, device=device)
    Xa = torch.cat([torch.ones((X.shape[0], 1), dtype=dtype, device=device), X], dim=1)

    Xsel = torch.as_tensor(X_select, dtype=dtype, device=device)
    ysel = torch.as_tensor(y_select.reshape(-1, 1), dtype=dtype, device=device)
    Xsel_a = torch.cat(
        [torch.ones((Xsel.shape[0], 1), dtype=dtype, device=device), Xsel],
        dim=1,
    )

    XtX = Xa.T @ Xa
    Xty = Xa.T @ y

    reg = torch.eye(Xa.shape[1], dtype=dtype, device=device)
    reg[0, 0] = 0.0  # no regularizar intercepto

    best_alpha = None
    best_sol = None
    best_score = float("inf")

    for alpha in alphas:
        A = XtX + float(alpha) * reg
        try:
            sol = torch.linalg.solve(A, Xty)
        except RuntimeError:
            sol = torch.linalg.pinv(A) @ Xty

        pred = Xsel_a @ sol
        score = torch.sqrt(torch.mean((pred - ysel) ** 2)).detach().cpu().item()

        if score < best_score:
            best_score = float(score)
            best_alpha = float(alpha)
            best_sol = sol.squeeze(1)

    if best_sol is None:
        raise RuntimeError("No se pudo ajustar Ridge con Torch.")

    return {
        "name": "RidgePoly2_torch",
        "backend": f"torch_{device.type}",
        "coef_standardized_poly": best_sol[1:].detach().cpu().numpy(),
        "intercept_standardized_poly": float(best_sol[0].detach().cpu().item()),
        "alpha": float(best_alpha),
        "selection_rmse": float(best_score),
    }


def fit_ridge_model(
    data: dict[str, Any],
    *,
    alphas: np.ndarray,
    backend: str,
    prefer_gpu: bool,
) -> dict[str, Any]:
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    # Para elegir alpha se usa val si existe. Si no, train.
    if len(data["val"]["y"]) > 0:
        X_select = data["val"]["X"]
        y_select = data["val"]["y"]
        alpha_selection_split = "val"
    else:
        X_select = X_train
        y_select = y_train
        alpha_selection_split = "train"

    if backend == "numpy":
        model = fit_ridge_numpy(
            X_train,
            y_train,
            X_select,
            y_select,
            alphas=alphas,
        )
    elif backend == "torch":
        model = fit_ridge_torch(
            X_train,
            y_train,
            X_select,
            y_select,
            alphas=alphas,
            prefer_gpu=prefer_gpu,
        )
    elif backend == "auto":
        if prefer_gpu:
            try:
                model = fit_ridge_torch(
                    X_train,
                    y_train,
                    X_select,
                    y_select,
                    alphas=alphas,
                    prefer_gpu=True,
                )
            except Exception as exc:
                print(
                    f"Aviso: no se pudo usar Torch/GPU ({exc}). Se usa NumPy/CPU.",
                    file=sys.stderr,
                )
                model = fit_ridge_numpy(
                    X_train,
                    y_train,
                    X_select,
                    y_select,
                    alphas=alphas,
                )
        else:
            model = fit_ridge_numpy(
                X_train,
                y_train,
                X_select,
                y_select,
                alphas=alphas,
            )
    else:
        raise ValueError("backend debe ser 'auto', 'numpy' o 'torch'.")

    model["alpha_selection_split"] = alpha_selection_split
    return model


def predict_ridge(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    return (
        float(model["intercept_standardized_poly"])
        + X @ np.asarray(model["coef_standardized_poly"], dtype=np.float64)
    )


# =============================================================================
# Tablas de salida
# =============================================================================


def build_metrics_table(data: dict[str, Any], model: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split_name in ["train", "val", "test"]:
        X = data[split_name]["X"]
        y = data[split_name]["y"]

        if len(y) == 0:
            met = compute_metrics(y, np.array([]))
        else:
            pred = predict_ridge(model, X)
            met = compute_metrics(y, pred)

        rows.append(
            {
                "model": model["name"],
                "split": split_name,
                "r2": met["r2"],
                "mae": met["mae"],
                "rmse": met["rmse"],
                "n": met["n"],
                "alpha": model["alpha"],
                "alpha_selection_split": model["alpha_selection_split"],
                "selection_rmse": model["selection_rmse"],
                "backend": model["backend"],
            }
        )

    return pd.DataFrame(rows)


def build_predictions_table(data: dict[str, Any], model: dict[str, Any]) -> pd.DataFrame:
    rows = []

    for split_name in ["train", "val", "test"]:
        split_data = data[split_name]
        X = split_data["X"]
        y = split_data["y"]
        df_split = split_data["df"].reset_index(drop=True)

        if len(df_split) == 0:
            continue

        pred = predict_ridge(model, X)

        base_cols = [c for c in META_COLS if c in df_split.columns]
        tmp = df_split[base_cols].copy()
        if "split" not in tmp.columns:
            tmp["split"] = split_name
        if "temperature" not in tmp.columns:
            tmp["temperature"] = y

        tmp["model"] = model["name"]
        tmp["y_true"] = y
        tmp["y_pred"] = pred
        tmp["error"] = pred - y
        tmp["abs_error"] = np.abs(pred - y)
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_coefficients_table(data: dict[str, Any], model: dict[str, Any]) -> pd.DataFrame:
    poly_names = data["poly_feature_names"]
    coef = np.asarray(model["coef_standardized_poly"], dtype=np.float64)

    coef_df = pd.DataFrame(
        {
            "model": model["name"],
            "poly_feature": poly_names,
            "coef_standardized_poly": coef,
            "abs_coef_standardized_poly": np.abs(coef),
            "alpha": model["alpha"],
            "backend": model["backend"],
        }
    ).sort_values("abs_coef_standardized_poly", ascending=False)

    intercept_row = pd.DataFrame(
        {
            "model": [model["name"]],
            "poly_feature": ["intercept"],
            "coef_standardized_poly": [float(model["intercept_standardized_poly"])],
            "abs_coef_standardized_poly": [abs(float(model["intercept_standardized_poly"]))],
            "alpha": [model["alpha"]],
            "backend": [model["backend"]],
        }
    )

    return pd.concat([intercept_row, coef_df], ignore_index=True)


def save_model_npz(
    output_dir: Path,
    data: dict[str, Any],
    model: dict[str, Any],
    feature_cols: list[str],
) -> None:
    """
    Guarda lo suficiente para reconstruir la prediccion sin depender de objetos sklearn.

    El flujo de prediccion es:
        X_raw
        -> imputar con imputer_statistics
        -> escalar con scaler_raw_mean/scale
        -> PolynomialFeatures con el mismo grado
        -> escalar con scaler_poly_mean/scale
        -> y = intercept + X_poly_scaled @ coef
    """
    imputer: SimpleImputer = data["imputer"]
    scaler_raw: StandardScaler = data["scaler_raw"]
    scaler_poly: StandardScaler = data["scaler_poly"]

    np.savez(
        output_dir / "modelo_ridge_polinomico_grado2.npz",
        feature_cols=np.asarray(feature_cols, dtype=object),
        poly_feature_names=np.asarray(data["poly_feature_names"], dtype=object),
        imputer_statistics=imputer.statistics_,
        scaler_raw_mean=scaler_raw.mean_,
        scaler_raw_scale=scaler_raw.scale_,
        scaler_poly_mean=scaler_poly.mean_,
        scaler_poly_scale=scaler_poly.scale_,
        coef_standardized_poly=np.asarray(model["coef_standardized_poly"], dtype=np.float64),
        intercept_standardized_poly=np.asarray([model["intercept_standardized_poly"]], dtype=np.float64),
        alpha=np.asarray([model["alpha"]], dtype=np.float64),
    )


# =============================================================================
# Figuras
# =============================================================================


def save_plots(output_dir: Path, predictions_df: pd.DataFrame, metrics_df: pd.DataFrame, coef_df: pd.DataFrame) -> None:
    if predictions_df.empty:
        return

    # Figura 1: prediccion vs real en test
    sub = predictions_df[predictions_df["split"] == "test"].copy()
    if len(sub) > 0:
        row = metrics_df[metrics_df["split"] == "test"].iloc[0]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sub["y_true"], sub["y_pred"], alpha=0.7, edgecolors="k", linewidths=0.3)

        lim_min = min(sub["y_true"].min(), sub["y_pred"].min()) - 1.0
        lim_max = max(sub["y_true"].max(), sub["y_pred"].max()) + 1.0
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", linewidth=1.2)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("Temperatura real (°C)")
        ax.set_ylabel("Temperatura predicha (°C)")
        ax.set_title(
            "Ridge polinomica grado 2 — test\n"
            f"R²={row['r2']:.4f}  RMSE={row['rmse']:.4f} °C  MAE={row['mae']:.4f} °C"
        )
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        fig.savefig(output_dir / "prediccion_vs_real_test_ridge_poly2.png", dpi=150)
        plt.close(fig)

        # Figura 2: residuos en test
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sub["y_true"], sub["error"], alpha=0.7, edgecolors="k", linewidths=0.3)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Temperatura real (°C)")
        ax.set_ylabel("Error prediccion - real (°C)")
        ax.set_title("Residuos en test — Ridge polinomica grado 2")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        fig.savefig(output_dir / "residuos_test_ridge_poly2.png", dpi=150)
        plt.close(fig)

    # Figura 3: top coeficientes polinomicos
    coef_no_intercept = coef_df[coef_df["poly_feature"] != "intercept"].copy()
    top = coef_no_intercept.head(25)
    if len(top) > 0:
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.barh(top["poly_feature"][::-1], top["coef_standardized_poly"][::-1])
        ax.axvline(0.0, linewidth=0.8)
        ax.set_xlabel("Coeficiente en espacio polinomico estandarizado")
        ax.set_title("Top 25 coeficientes — Ridge polinomica grado 2")
        ax.tick_params(axis="y", labelsize=7)
        plt.tight_layout()
        fig.savefig(output_dir / "coeficientes_top25_ridge_poly2.png", dpi=150)
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    csv_default, output_default = default_paths()

    parser = argparse.ArgumentParser(
        description=(
            "Regresion Ridge polinomica de grado 2 a partir de "
            "dataset_features_temperatura.csv respetando los splits train/val/test."
        )
    )
    parser.add_argument("--csv-path", type=Path, default=csv_default)
    parser.add_argument("--output-dir", type=Path, default=output_default)
    parser.add_argument("--features", type=str, default=None, help="Lista separada por comas. Si se omite, usa las 9 features seleccionadas.")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--interaction-only", action="store_true", help="Usa solo interacciones, sin cuadrados.")
    parser.add_argument("--alpha-log-min", type=float, default=-4.0)
    parser.add_argument("--alpha-log-max", type=float, default=6.0)
    parser.add_argument("--n-alphas", type=int, default=200)
    parser.add_argument("--alphas", type=str, default=None, help="Lista explicita de alphas separada por comas.")
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--prefer-gpu", action="store_true", help="Si se usa Torch y hay CUDA, entrena en GPU.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-excel", action="store_true")

    args = parser.parse_args()

    feature_cols = parse_feature_list(args.features)
    alphas = parse_alphas(args)

    if args.degree < 1:
        raise ValueError("--degree debe ser >= 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.csv_path, feature_cols)
    data = prepare_arrays(
        df,
        feature_cols,
        degree=args.degree,
        interaction_only=args.interaction_only,
    )

    model = fit_ridge_model(
        data,
        alphas=alphas,
        backend=args.backend,
        prefer_gpu=args.prefer_gpu,
    )

    metrics_df = build_metrics_table(data, model)
    predictions_df = build_predictions_table(data, model)
    coef_df = build_coefficients_table(data, model)

    # Guardados principales
    metrics_path = args.output_dir / "metricas_ridge_polinomica_grado2.csv"
    predictions_path = args.output_dir / "predicciones_ridge_polinomica_grado2.csv"
    coef_path = args.output_dir / "coeficientes_ridge_polinomica_grado2.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    coef_df.to_csv(coef_path, index=False)

    # Guardar modelo y configuracion
    save_model_npz(args.output_dir, data, model, feature_cols)

    config = {
        "csv_path": str(args.csv_path),
        "output_dir": str(args.output_dir),
        "selected_features": feature_cols,
        "degree": args.degree,
        "interaction_only": args.interaction_only,
        "n_original_features": len(feature_cols),
        "n_polynomial_features": len(data["poly_feature_names"]),
        "alphas": alphas.tolist(),
        "best_alpha": model["alpha"],
        "alpha_selection_split": model["alpha_selection_split"],
        "selection_rmse": model["selection_rmse"],
        "backend": model["backend"],
    }
    with open(args.output_dir / "config_ridge_polinomica_grado2.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(args.output_dir / "features_polinomicas_grado2.json", "w", encoding="utf-8") as f:
        json.dump(data["poly_feature_names"], f, indent=2, ensure_ascii=False)

    if not args.no_excel:
        excel_path = args.output_dir / "resultados_ridge_polinomica_grado2.xlsx"
        try:
            with pd.ExcelWriter(excel_path) as writer:
                metrics_df.to_excel(writer, sheet_name="metricas", index=False)
                coef_df.to_excel(writer, sheet_name="coeficientes", index=False)
                predictions_df.to_excel(writer, sheet_name="predicciones", index=False)
        except Exception as exc:
            print(f"Aviso: no se pudo guardar Excel ({exc}). Se mantienen los CSV.", file=sys.stderr)

    if not args.no_plots:
        save_plots(args.output_dir, predictions_df, metrics_df, coef_df)

    # Salida por pantalla
    print("\n=== Ridge polinomica ===")
    print(f"CSV: {args.csv_path}")
    print(f"Features originales: {len(feature_cols)}")
    print(f"Features polinomicas: {len(data['poly_feature_names'])}")
    print(f"Backend: {model['backend']}")
    print(f"Alpha elegido: {model['alpha']:.6g}")
    print(f"Alpha elegido usando split: {model['alpha_selection_split']}")
    print(f"RMSE de seleccion: {model['selection_rmse']:.6f} °C")

    print("\nMetricas:")
    print(metrics_df[["split", "r2", "mae", "rmse", "n"]].to_string(index=False))

    print("\nTop 20 coeficientes polinomicos por valor absoluto:")
    top20 = coef_df[coef_df["poly_feature"] != "intercept"].head(20)
    print(top20[["poly_feature", "coef_standardized_poly", "abs_coef_standardized_poly"]].to_string(index=False))

    print("\nArchivos guardados en:")
    print(f"  {args.output_dir}")


if __name__ == "__main__":
    main()
