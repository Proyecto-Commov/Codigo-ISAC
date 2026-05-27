from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LassoCV, Lasso
import matplotlib.pyplot as plt


# =============================================================================
# Configuracion
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
    """
    Supone que el script se ejecuta/guarda en Codigo-ISAC/Hugo y que el CSV esta en:

        Codigo-ISAC/Hugo/dataset_features_temperatura/dataset_features_temperatura.csv
    """
    csv_path = THIS_DIR / "dataset_features_temperatura" / "dataset_features_temperatura.csv"
    output_dir = THIS_DIR / "resultados_regresion_features_seleccionadas"
    return csv_path, output_dir


# =============================================================================
# Lectura y preparacion de datos
# =============================================================================

def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["split", "temperature", *SELECTED_FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias en el CSV:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    df = df.copy()
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")

    for c in SELECTED_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["temperature"].notna()].copy()

    valid_splits = {"train", "val", "test"}
    bad_splits = sorted(set(df["split"].dropna()) - valid_splits)
    if bad_splits:
        raise ValueError(f"Hay valores de split no reconocidos: {bad_splits}")

    if not {"train", "test"}.issubset(set(df["split"])):
        raise ValueError("El CSV debe contener al menos muestras con split='train' y split='test'.")

    return df


def prepare_arrays(df: pd.DataFrame) -> dict[str, Any]:
    """
    Usa exclusivamente el split ya definido en el CSV.

    La imputacion y la estandarizacion se ajustan SOLO con train.
    Luego se aplican a val y test.
    """
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if len(train_df) == 0:
        raise ValueError("No hay muestras train.")
    if len(test_df) == 0:
        raise ValueError("No hay muestras test.")

    X_train_raw = train_df[SELECTED_FEATURES]
    y_train = train_df["temperature"].to_numpy(dtype=np.float64)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train_raw)
    X_train = scaler.fit_transform(X_train_imp).astype(np.float64)

    out: dict[str, Any] = {
        "train": {
            "df": train_df,
            "X": X_train,
            "y": y_train,
        },
        "imputer": imputer,
        "scaler": scaler,
    }

    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        if len(split_df) == 0:
            out[split_name] = {
                "df": split_df,
                "X": np.empty((0, len(SELECTED_FEATURES)), dtype=np.float64),
                "y": np.empty((0,), dtype=np.float64),
            }
            continue

        X_imp = imputer.transform(split_df[SELECTED_FEATURES])
        X = scaler.transform(X_imp).astype(np.float64)
        y = split_df["temperature"].to_numpy(dtype=np.float64)

        out[split_name] = {
            "df": split_df,
            "X": X,
            "y": y,
        }

    return out


# =============================================================================
# Metricas
# =============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan, "n": 0}

    return {
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "n": int(len(y_true)),
    }


# =============================================================================
# Backend Torch para OLS/Ridge
# =============================================================================

def get_torch_device(prefer_gpu: bool):
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def torch_fit_ols(X_train: np.ndarray, y_train: np.ndarray, *, prefer_gpu: bool) -> dict[str, Any]:
    import torch

    device = get_torch_device(prefer_gpu)
    dtype = torch.float64

    X = torch.as_tensor(X_train, dtype=dtype, device=device)
    y = torch.as_tensor(y_train.reshape(-1, 1), dtype=dtype, device=device)
    ones = torch.ones((X.shape[0], 1), dtype=dtype, device=device)
    Xa = torch.cat([ones, X], dim=1)

    # Solucion de minimos cuadrados: y = b0 + X beta
    sol = torch.linalg.lstsq(Xa, y).solution.squeeze(1)

    return {
        "name": "OLS_torch",
        "backend": f"torch_{device.type}",
        "coef_standardized": sol[1:].detach().cpu().numpy(),
        "intercept_standardized": float(sol[0].detach().cpu().item()),
        "alpha": np.nan,
    }


def torch_fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    alphas: np.ndarray,
    prefer_gpu: bool,
) -> dict[str, Any]:
    import torch

    device = get_torch_device(prefer_gpu)
    dtype = torch.float64

    X = torch.as_tensor(X_train, dtype=dtype, device=device)
    y = torch.as_tensor(y_train.reshape(-1, 1), dtype=dtype, device=device)
    ones = torch.ones((X.shape[0], 1), dtype=dtype, device=device)
    Xa = torch.cat([ones, X], dim=1)

    XtX = Xa.T @ Xa
    Xty = Xa.T @ y

    # No regularizar el intercepto.
    reg = torch.eye(Xa.shape[1], dtype=dtype, device=device)
    reg[0, 0] = 0.0

    best_alpha = None
    best_sol = None
    best_score = np.inf

    # Si hay validacion, se elige alpha por RMSE en val.
    # Si no hay val, se elige por RMSE en train.
    if len(X_val) > 0:
        Xsel_np = X_val
        ysel_np = y_val
    else:
        Xsel_np = X_train
        ysel_np = y_train

    Xsel = torch.as_tensor(Xsel_np, dtype=dtype, device=device)
    ysel = torch.as_tensor(ysel_np.reshape(-1, 1), dtype=dtype, device=device)
    Xsel_a = torch.cat([torch.ones((Xsel.shape[0], 1), dtype=dtype, device=device), Xsel], dim=1)

    for alpha in alphas:
        A = XtX + float(alpha) * reg
        sol = torch.linalg.solve(A, Xty)
        pred = Xsel_a @ sol
        score = torch.sqrt(torch.mean((pred - ysel) ** 2)).detach().cpu().item()

        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
            best_sol = sol.squeeze(1)

    assert best_sol is not None

    return {
        "name": "Ridge_torch",
        "backend": f"torch_{device.type}",
        "coef_standardized": best_sol[1:].detach().cpu().numpy(),
        "intercept_standardized": float(best_sol[0].detach().cpu().item()),
        "alpha": float(best_alpha),
    }


def predict_linear(model: dict[str, Any], X: np.ndarray) -> np.ndarray:
    return model["intercept_standardized"] + X @ model["coef_standardized"]


# =============================================================================
# Lasso CPU opcional
# =============================================================================

def fit_lasso_cpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    alphas: np.ndarray,
    random_state: int,
) -> dict[str, Any]:
    """
    Lasso no se implementa en GPU aqui porque scikit-learn trabaja en CPU.
    Se incluye como comparacion si se solicita.
    """
    if len(X_val) > 0:
        best_alpha = None
        best_model = None
        best_score = np.inf

        for alpha in alphas:
            model = Lasso(alpha=float(alpha), max_iter=200000, random_state=random_state)
            model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            score = rmse(y_val, pred_val)
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
                best_model = model

        assert best_model is not None
        m = best_model
    else:
        m = LassoCV(
            alphas=alphas,
            cv=5,
            max_iter=200000,
            random_state=random_state,
        )
        m.fit(X_train, y_train)
        best_alpha = float(m.alpha_)

    return {
        "name": "Lasso_sklearn",
        "backend": "sklearn_cpu",
        "coef_standardized": np.asarray(m.coef_, dtype=float),
        "intercept_standardized": float(m.intercept_),
        "alpha": float(best_alpha),
    }


# =============================================================================
# Coeficientes en escala original
# =============================================================================

def coefficients_dataframe(model: dict[str, Any], scaler: StandardScaler) -> pd.DataFrame:
    coef_std = np.asarray(model["coef_standardized"], dtype=float)
    coef_original = coef_std / scaler.scale_
    intercept_original = float(model["intercept_standardized"] - np.sum(coef_std * scaler.mean_ / scaler.scale_))

    df_coef = pd.DataFrame(
        {
            "model": model["name"],
            "feature": SELECTED_FEATURES,
            "coef_standardized": coef_std,
            "coef_original_units": coef_original,
            "abs_coef_standardized": np.abs(coef_std),
            "alpha": model.get("alpha", np.nan),
            "backend": model.get("backend", ""),
        }
    ).sort_values("abs_coef_standardized", ascending=False)

    intercept_row = pd.DataFrame(
        {
            "model": [model["name"]],
            "feature": ["intercept"],
            "coef_standardized": [model["intercept_standardized"]],
            "coef_original_units": [intercept_original],
            "abs_coef_standardized": [abs(model["intercept_standardized"])],
            "alpha": [model.get("alpha", np.nan)],
            "backend": [model.get("backend", "")],
        }
    )

    return pd.concat([intercept_row, df_coef], ignore_index=True)


# =============================================================================
# Salidas
# =============================================================================

def build_predictions_table(data: dict[str, Any], models: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for split_name in ["train", "val", "test"]:
        split_data = data[split_name]
        X = split_data["X"]
        y = split_data["y"]
        df_split = split_data["df"].reset_index(drop=True)

        if len(df_split) == 0:
            continue

        base_cols = [c for c in META_COLS if c in df_split.columns]
        base = df_split[base_cols].copy()
        if "split" not in base.columns:
            base["split"] = split_name
        if "temperature" not in base.columns:
            base["temperature"] = y

        for model in models:
            pred = predict_linear(model, X)
            tmp = base.copy()
            tmp["model"] = model["name"]
            tmp["y_true"] = y
            tmp["y_pred"] = pred
            tmp["error"] = pred - y
            tmp["abs_error"] = np.abs(pred - y)
            rows.append(tmp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_metrics_table(data: dict[str, Any], models: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for model in models:
        for split_name in ["train", "val", "test"]:
            X = data[split_name]["X"]
            y = data[split_name]["y"]

            if len(y) == 0:
                met = compute_metrics(y, np.array([]))
            else:
                pred = predict_linear(model, X)
                met = compute_metrics(y, pred)

            rows.append(
                {
                    "model": model["name"],
                    "split": split_name,
                    "r2": met["r2"],
                    "mae": met["mae"],
                    "rmse": met["rmse"],
                    "n": met["n"],
                    "alpha": model.get("alpha", np.nan),
                    "backend": model.get("backend", ""),
                }
            )

    return pd.DataFrame(rows)


def save_plots(output_dir: Path, predictions_df: pd.DataFrame, metrics_df: pd.DataFrame, coef_df: pd.DataFrame) -> None:
    if predictions_df.empty:
        return

    models = list(predictions_df["model"].unique())

    # Figura 1: prediccion vs real para test.
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = predictions_df[(predictions_df["model"] == model) & (predictions_df["split"] == "test")]
        ax.scatter(sub["y_true"], sub["y_pred"], alpha=0.65, edgecolors="k", linewidths=0.3)
        if len(sub) > 0:
            lo = min(sub["y_true"].min(), sub["y_pred"].min())
            hi = max(sub["y_true"].max(), sub["y_pred"].max())
            pad = 0.05 * (hi - lo + 1e-12)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", linewidth=1.2)

        row = metrics_df[(metrics_df["model"] == model) & (metrics_df["split"] == "test")]
        if len(row) == 1:
            r2 = row["r2"].iloc[0]
            rmse_v = row["rmse"].iloc[0]
            title = f"{model}\nR²={r2:.4f} | RMSE={rmse_v:.4f} °C"
        else:
            title = model

        ax.set_title(title)
        ax.set_xlabel("Temperatura real [°C]")
        ax.set_ylabel("Temperatura predicha [°C]")

    fig.tight_layout()
    fig.savefig(output_dir / "prediccion_vs_real_test.png", dpi=150)
    plt.close(fig)

    # Figura 2: coeficientes estandarizados.
    for model in models:
        sub = coef_df[(coef_df["model"] == model) & (coef_df["feature"] != "intercept")].copy()
        sub = sub.sort_values("coef_standardized")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(sub["feature"], sub["coef_standardized"])
        ax.axvline(0, linewidth=0.8)
        ax.set_title(f"Coeficientes estandarizados — {model}")
        ax.set_xlabel("Coeficiente")
        fig.tight_layout()
        safe_model = model.replace("/", "_").replace(" ", "_")
        fig.savefig(output_dir / f"coeficientes_{safe_model}.png", dpi=150)
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    default_csv, default_output = default_paths()

    parser = argparse.ArgumentParser(
        description=(
            "Regresion lineal de temperatura leyendo directamente "
            "dataset_features_temperatura.csv y usando 9 features seleccionadas."
        )
    )

    parser.add_argument("--csv-path", type=Path, default=default_csv)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--models", type=str, default="ols,ridge,lasso",
                        help="Modelos a entrenar separados por comas: ols,ridge,lasso")
    parser.add_argument("--prefer-gpu", action="store_true",
                        help="Usa CUDA para OLS/Ridge si PyTorch detecta GPU.")
    parser.add_argument("--ridge-alpha-min", type=float, default=1e-4)
    parser.add_argument("--ridge-alpha-max", type=float, default=1e6)
    parser.add_argument("--ridge-alpha-num", type=int, default=200)
    parser.add_argument("--lasso-alpha-min", type=float, default=1e-5)
    parser.add_argument("--lasso-alpha-max", type=float, default=1e2)
    parser.add_argument("--lasso-alpha-num", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-excel", action="store_true")

    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.csv_path)
    data = prepare_arrays(df)

    requested_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    valid_models = {"ols", "ridge", "lasso"}
    unknown = sorted(set(requested_models) - valid_models)
    if unknown:
        raise ValueError(f"Modelos no reconocidos: {unknown}")

    X_train = data["train"]["X"]
    y_train = data["train"]["y"]
    X_val = data["val"]["X"]
    y_val = data["val"]["y"]

    models: list[dict[str, Any]] = []

    if "ols" in requested_models:
        models.append(torch_fit_ols(X_train, y_train, prefer_gpu=args.prefer_gpu))

    if "ridge" in requested_models:
        ridge_alphas = np.logspace(
            np.log10(args.ridge_alpha_min),
            np.log10(args.ridge_alpha_max),
            int(args.ridge_alpha_num),
        )
        models.append(
            torch_fit_ridge(
                X_train,
                y_train,
                X_val,
                y_val,
                alphas=ridge_alphas,
                prefer_gpu=args.prefer_gpu,
            )
        )

    if "lasso" in requested_models:
        lasso_alphas = np.logspace(
            np.log10(args.lasso_alpha_min),
            np.log10(args.lasso_alpha_max),
            int(args.lasso_alpha_num),
        )
        models.append(
            fit_lasso_cpu(
                X_train,
                y_train,
                X_val,
                y_val,
                alphas=lasso_alphas,
                random_state=args.random_state,
            )
        )

    metrics_df = build_metrics_table(data, models)
    predictions_df = build_predictions_table(data, models)
    coef_df = pd.concat(
        [coefficients_dataframe(m, data["scaler"]) for m in models],
        ignore_index=True,
    )

    # Guardado principal.
    metrics_path = output_dir / "metricas_regresion_features_seleccionadas.csv"
    pred_path = output_dir / "predicciones_regresion_features_seleccionadas.csv"
    coef_path = output_dir / "coeficientes_regresion_features_seleccionadas.csv"
    features_path = output_dir / "features_usadas.json"
    config_path = output_dir / "config_regresion_features_seleccionadas.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(pred_path, index=False)
    coef_df.to_csv(coef_path, index=False)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(SELECTED_FEATURES, f, ensure_ascii=False, indent=2)

    config = {
        "csv_path": str(args.csv_path),
        "output_dir": str(output_dir),
        "selected_features": SELECTED_FEATURES,
        "models": requested_models,
        "prefer_gpu": bool(args.prefer_gpu),
        "splits": {s: int(len(data[s]["y"])) for s in ["train", "val", "test"]},
        "note": "Imputer y StandardScaler ajustados solo con train. Val se usa para seleccionar alpha en Ridge/Lasso si existe.",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if not args.no_excel:
        excel_path = output_dir / "regresion_features_seleccionadas.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            metrics_df.to_excel(writer, sheet_name="metricas", index=False)
            coef_df.to_excel(writer, sheet_name="coeficientes", index=False)
            predictions_df.to_excel(writer, sheet_name="predicciones", index=False)
            pd.DataFrame({"feature": SELECTED_FEATURES}).to_excel(writer, sheet_name="features_usadas", index=False)

    if not args.no_plots:
        save_plots(output_dir, predictions_df, metrics_df, coef_df)

    # Salida por pantalla.
    print("\nFeatures usadas:")
    for f in SELECTED_FEATURES:
        print(f"  - {f}")

    print("\nMuestras por split:")
    for split_name in ["train", "val", "test"]:
        print(f"  {split_name}: {len(data[split_name]['y'])}")

    print("\nMetricas:")
    print(metrics_df.to_string(index=False))

    print("\nCoeficientes guardados en:")
    print(f"  {coef_path}")
    print("Predicciones guardadas en:")
    print(f"  {pred_path}")
    print("Metricas guardadas en:")
    print(f"  {metrics_path}")

    print("\nBackends usados:")
    for m in models:
        print(f"  {m['name']}: {m['backend']}")


if __name__ == "__main__":
    main()
