from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# =============================================================================
# Imports locales del proyecto
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from channel_estimator import compute_channel_matrix_from_iq_paths
    from channel_features_complete import extract_channel_matrix_features
except ImportError as exc:
    raise ImportError(
        "No se han podido importar channel_estimator.py y/o "
        "channel_features_complete.py. Guarda este script en la misma carpeta "
        "Hugo donde están esos archivos, o añade esa carpeta al PYTHONPATH."
    ) from exc


# =============================================================================
# Rutas por defecto
# =============================================================================

def default_yaml_path() -> Path:
    # Esperado si el script está en Codigo-ISAC/Hugo:
    # Codigo-ISAC/data/Modulator.yaml
    return THIS_DIR.parent / "data" / "Modulator.yaml"


def default_model_path() -> Path:
    return THIS_DIR / "resultados_ridge_polinomica_grado2" / "modelo_ridge_polinomico_grado2.npz"


def default_output_dir() -> Path:
    return THIS_DIR / "inferencias_ridge_polinomico"


# =============================================================================
# Modelo Ridge polinómico guardado en .npz
# =============================================================================

def _as_str_list(arr: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(arr).tolist()]


def load_ridge_poly_model(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {model_path}\n"
            "Primero debes ejecutar regresion_ridge_polinomica_features_csv.py "
            "para generar modelo_ridge_polinomico_grado2.npz."
        )

    data = np.load(model_path, allow_pickle=True)

    required_keys = [
        "feature_cols",
        "poly_feature_names",
        "imputer_statistics",
        "scaler_raw_mean",
        "scaler_raw_scale",
        "scaler_poly_mean",
        "scaler_poly_scale",
        "coef_standardized_poly",
        "intercept_standardized_poly",
        "alpha",
    ]
    missing = [k for k in required_keys if k not in data.files]
    if missing:
        raise ValueError(
            "El .npz del modelo no contiene todas las claves necesarias:\n"
            + "\n".join(f"  - {k}" for k in missing)
        )

    model = {
        "feature_cols": _as_str_list(data["feature_cols"]),
        "poly_feature_names": _as_str_list(data["poly_feature_names"]),
        "imputer_statistics": np.asarray(data["imputer_statistics"], dtype=np.float64),
        "scaler_raw_mean": np.asarray(data["scaler_raw_mean"], dtype=np.float64),
        "scaler_raw_scale": np.asarray(data["scaler_raw_scale"], dtype=np.float64),
        "scaler_poly_mean": np.asarray(data["scaler_poly_mean"], dtype=np.float64),
        "scaler_poly_scale": np.asarray(data["scaler_poly_scale"], dtype=np.float64),
        "coef_standardized_poly": np.asarray(data["coef_standardized_poly"], dtype=np.float64),
        "intercept_standardized_poly": float(np.asarray(data["intercept_standardized_poly"], dtype=np.float64).ravel()[0]),
        "alpha": float(np.asarray(data["alpha"], dtype=np.float64).ravel()[0]),
        "model_path": str(model_path),
    }

    n_features = len(model["feature_cols"])
    n_poly = len(model["poly_feature_names"])

    shape_checks = {
        "imputer_statistics": len(model["imputer_statistics"]),
        "scaler_raw_mean": len(model["scaler_raw_mean"]),
        "scaler_raw_scale": len(model["scaler_raw_scale"]),
    }
    bad_raw = {k: v for k, v in shape_checks.items() if v != n_features}
    if bad_raw:
        raise ValueError(
            "Inconsistencia en el modelo: vectores de preprocesado con "
            f"dimensión distinta de n_features={n_features}: {bad_raw}"
        )

    shape_checks_poly = {
        "scaler_poly_mean": len(model["scaler_poly_mean"]),
        "scaler_poly_scale": len(model["scaler_poly_scale"]),
        "coef_standardized_poly": len(model["coef_standardized_poly"]),
    }
    bad_poly = {k: v for k, v in shape_checks_poly.items() if v != n_poly}
    if bad_poly:
        raise ValueError(
            "Inconsistencia en el modelo: vectores polinómicos con "
            f"dimensión distinta de n_poly={n_poly}: {bad_poly}"
        )

    return model


# =============================================================================
# Extracción de features desde TX/RX
# =============================================================================

def compute_features_from_iq_pair(
    tx_path: Path,
    rx_path: Path,
    yaml_path: Path,
    *,
    n_symbols: int | None,
    start_sample_tx: int,
    start_sample_rx: int,
    storage_format_tx: str,
    storage_format_rx: str,
    fftshift: bool,
    normalize_fft: bool,
    eps: float,
    trim_to_complete_frames: bool,
    entropy_bins: int,
    verbose: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    1) Lee TX/RX .bin.
    2) Demodula OFDM.
    3) Calcula H[m,k] = Y[m,k] / X[m,k].
    4) Extrae las características de channel_features_complete.py.
    """
    aux = compute_channel_matrix_from_iq_paths(
        tx_path,
        rx_path,
        yaml_path,
        n_symbols=n_symbols,
        start_sample_tx=start_sample_tx,
        start_sample_rx=start_sample_rx,
        storage_format_tx=storage_format_tx,
        storage_format_rx=storage_format_rx,
        fftshift=fftshift,
        normalize_fft=normalize_fft,
        eps=eps,
        trim_to_complete_frames=trim_to_complete_frames,
        return_aux=True,
        output_order="mk",
        verbose=verbose,
    )

    H = aux["H"]
    features = extract_channel_matrix_features(
        H,
        input_order="mk",
        entropy_bins=entropy_bins,
        eps=eps,
    )

    return features, aux


# =============================================================================
# Predicción
# =============================================================================

def build_raw_feature_vector(features: dict[str, float], feature_cols: list[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in features]
    if missing:
        raise ValueError(
            "El modelo pide features que no aparecen en channel_features_complete.py:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    x = np.array([features[c] for c in feature_cols], dtype=np.float64).reshape(1, -1)
    x[~np.isfinite(x)] = np.nan
    return x


def apply_saved_preprocessing_and_predict(
    x_raw: np.ndarray,
    model: dict[str, Any],
    *,
    degree: int,
    interaction_only: bool,
) -> tuple[float, dict[str, np.ndarray | list[str]]]:
    """
    Reproduce exactamente el flujo guardado por regresion_ridge_polinomica_features_csv.py:

        X_raw
        -> imputación con medianas de train
        -> escalado de features originales
        -> expansión polinómica
        -> escalado de features polinómicas
        -> y = intercept + X_poly_scaled @ coef
    """
    feature_cols = model["feature_cols"]

    stats = model["imputer_statistics"]
    if np.any(~np.isfinite(stats)):
        bad = [feature_cols[i] for i, v in enumerate(stats) if not np.isfinite(v)]
        raise ValueError(
            "El modelo contiene estadísticas de imputación no finitas para:\n"
            + "\n".join(f"  - {c}" for c in bad)
        )

    # Imputación tipo SimpleImputer(strategy="median") para una sola muestra.
    x_imp = x_raw.copy()
    nan_mask = np.isnan(x_imp)
    if np.any(nan_mask):
        x_imp[nan_mask] = np.take(stats, np.where(nan_mask)[1])

    # Escalado de variables originales.
    raw_scale = model["scaler_raw_scale"].copy()
    raw_scale[raw_scale == 0.0] = 1.0
    x_sc = (x_imp - model["scaler_raw_mean"]) / raw_scale

    # Expansión polinómica.
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False,
        interaction_only=interaction_only,
    )
    x_poly = poly.fit_transform(x_sc)
    computed_poly_names = list(poly.get_feature_names_out(feature_cols))

    saved_poly_names = model["poly_feature_names"]
    if len(computed_poly_names) != len(saved_poly_names):
        raise ValueError(
            "La expansión polinómica calculada no coincide con el modelo guardado.\n"
            f"Calculadas: {len(computed_poly_names)} features polinómicas.\n"
            f"Modelo:     {len(saved_poly_names)} features polinómicas.\n"
            "Comprueba --degree y --interaction-only."
        )

    # Normalmente deberían coincidir exactamente.
    if computed_poly_names != saved_poly_names:
        # Si no coinciden, no conviene seguir: el orden de coeficientes podría ser incorrecto.
        first_bad = next(
            (i for i, (a, b) in enumerate(zip(computed_poly_names, saved_poly_names)) if a != b),
            None,
        )
        raise ValueError(
            "Los nombres/orden de las features polinómicas no coinciden con el modelo.\n"
            f"Primera diferencia en posición {first_bad}:\n"
            f"  calculada: {computed_poly_names[first_bad]}\n"
            f"  modelo:    {saved_poly_names[first_bad]}\n"
        )

    # Escalado de features polinómicas.
    poly_scale = model["scaler_poly_scale"].copy()
    poly_scale[poly_scale == 0.0] = 1.0
    x_poly_sc = (x_poly - model["scaler_poly_mean"]) / poly_scale

    y_pred = (
        model["intercept_standardized_poly"]
        + x_poly_sc @ model["coef_standardized_poly"]
    )

    details = {
        "x_raw": x_raw.ravel(),
        "x_imputed": x_imp.ravel(),
        "x_scaled": x_sc.ravel(),
        "x_poly_scaled": x_poly_sc.ravel(),
        "computed_poly_names": computed_poly_names,
    }
    return float(y_pred.ravel()[0]), details


# =============================================================================
# Guardado de resultados
# =============================================================================

def save_outputs(
    output_dir: Path,
    *,
    tx_path: Path,
    rx_path: Path,
    yaml_path: Path,
    model: dict[str, Any],
    temperature_pred: float,
    features: dict[str, float],
    aux: dict[str, Any],
    raw_vector: np.ndarray,
    details: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tabla de features completa.
    features_row = {
        "tx_path": str(tx_path),
        "rx_path": str(rx_path),
        "yaml_path": str(yaml_path),
        "model_path": str(args.model_path),
        "temperature_pred": temperature_pred,
        "n_symbols_used": aux.get("n_symbols_used", None),
        "fs_slow": aux.get("fs_slow", None),
        "symbol_duration": aux.get("symbol_duration", None),
        **features,
    }
    pd.DataFrame([features_row]).to_csv(output_dir / "features_y_prediccion.csv", index=False)

    # Vector usado por el modelo.
    used_df = pd.DataFrame(
        {
            "feature": model["feature_cols"],
            "value_raw": raw_vector.ravel(),
            "value_imputed": details["x_imputed"],
            "value_scaled": details["x_scaled"],
        }
    )
    used_df.to_csv(output_dir / "features_usadas_por_modelo.csv", index=False)

    # Contribuciones en espacio polinómico estandarizado.
    coef = model["coef_standardized_poly"]
    xpoly = details["x_poly_scaled"]
    contributions = xpoly * coef
    contrib_df = pd.DataFrame(
        {
            "poly_feature": model["poly_feature_names"],
            "x_poly_scaled": xpoly,
            "coef_standardized_poly": coef,
            "contribution": contributions,
            "abs_contribution": np.abs(contributions),
        }
    ).sort_values("abs_contribution", ascending=False)
    contrib_df.to_csv(output_dir / "contribuciones_polinomicas.csv", index=False)

    # JSON resumen.
    summary = {
        "temperature_pred": temperature_pred,
        "unit": "degC",
        "tx_path": str(tx_path),
        "rx_path": str(rx_path),
        "yaml_path": str(yaml_path),
        "model_path": str(args.model_path),
        "alpha": model["alpha"],
        "n_original_features_model": len(model["feature_cols"]),
        "n_polynomial_features_model": len(model["poly_feature_names"]),
        "degree": args.degree,
        "interaction_only": args.interaction_only,
        "n_symbols_used": aux.get("n_symbols_used", None),
        "fs_slow": aux.get("fs_slow", None),
        "symbol_duration": aux.get("symbol_duration", None),
        "storage_format_tx": args.storage_format_tx,
        "storage_format_rx": args.storage_format_rx,
    }
    with open(output_dir / "prediccion_temperatura.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inferencia final de temperatura: TX/RX .bin -> H -> features -> "
            "Ridge polinómico entrenado."
        )
    )

    parser.add_argument("--tx-path", type=Path, required=True, help="Ruta al archivo iq_tx_*.bin.")
    parser.add_argument("--rx-path", type=Path, required=True, help="Ruta al archivo iq_rx_*.bin.")
    parser.add_argument("--yaml-path", type=Path, default=default_yaml_path(), help="Ruta a Modulator.yaml.")
    parser.add_argument("--model-path", type=Path, default=default_model_path(), help="Ruta a modelo_ridge_polinomico_grado2.npz.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Carpeta donde guardar la inferencia.")

    parser.add_argument("--n-symbols", type=int, default=None, help="Número de símbolos OFDM a usar. Por defecto usa los comunes disponibles.")
    parser.add_argument("--start-sample-tx", type=int, default=0)
    parser.add_argument("--start-sample-rx", type=int, default=0)
    parser.add_argument(
        "--storage-format-tx",
        choices=["auto", "complex64", "sc16_interleaved", "sc8_interleaved"],
        default="auto",
    )
    parser.add_argument(
        "--storage-format-rx",
        choices=["auto", "complex64", "sc16_interleaved", "sc8_interleaved"],
        default="auto",
    )
    parser.add_argument("--fftshift", action="store_true")
    parser.add_argument("--normalize-fft", action="store_true")
    parser.add_argument("--no-trim", action="store_true", help="No recorta a tramas completas antes de demodular.")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--entropy-bins", type=int, default=64)

    parser.add_argument("--degree", type=int, default=2, help="Debe coincidir con el entrenamiento. Por defecto: 2.")
    parser.add_argument("--interaction-only", action="store_true", help="Debe coincidir con el entrenamiento si se usó.")

    parser.add_argument("--quiet", action="store_true", help="Reduce la salida por pantalla del cálculo de H.")
    parser.add_argument("--no-save", action="store_true", help="No guarda CSV/JSON de salida; solo imprime la predicción.")

    args = parser.parse_args()

    if args.degree < 1:
        raise ValueError("--degree debe ser >= 1.")

    model = load_ridge_poly_model(args.model_path)

    features, aux = compute_features_from_iq_pair(
        args.tx_path,
        args.rx_path,
        args.yaml_path,
        n_symbols=args.n_symbols,
        start_sample_tx=args.start_sample_tx,
        start_sample_rx=args.start_sample_rx,
        storage_format_tx=args.storage_format_tx,
        storage_format_rx=args.storage_format_rx,
        fftshift=args.fftshift,
        normalize_fft=args.normalize_fft,
        eps=args.eps,
        trim_to_complete_frames=not args.no_trim,
        entropy_bins=args.entropy_bins,
        verbose=not args.quiet,
    )

    x_raw = build_raw_feature_vector(features, model["feature_cols"])
    temperature_pred, details = apply_saved_preprocessing_and_predict(
        x_raw,
        model,
        degree=args.degree,
        interaction_only=args.interaction_only,
    )

    print("\n=== Predicción de temperatura ===")
    print(f"TX: {args.tx_path}")
    print(f"RX: {args.rx_path}")
    print(f"Modelo: {args.model_path}")
    print(f"Features originales usadas por el modelo: {len(model['feature_cols'])}")
    print(f"Features polinómicas usadas por el modelo: {len(model['poly_feature_names'])}")
    print(f"Alpha Ridge: {model['alpha']:.6g}")
    print(f"Temperatura predicha: {temperature_pred:.6f} °C")

    if not args.no_save:
        save_outputs(
            args.output_dir,
            tx_path=args.tx_path,
            rx_path=args.rx_path,
            yaml_path=args.yaml_path,
            model=model,
            temperature_pred=temperature_pred,
            features=features,
            aux=aux,
            raw_vector=x_raw,
            details=details,
            args=args,
        )
        print("\nArchivos guardados en:")
        print(f"  {args.output_dir}")
        print("  - prediccion_temperatura.json")
        print("  - features_y_prediccion.csv")
        print("  - features_usadas_por_modelo.csv")
        print("  - contribuciones_polinomicas.csv")


if __name__ == "__main__":
    main()
