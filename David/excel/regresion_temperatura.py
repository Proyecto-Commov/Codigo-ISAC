import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=ConvergenceWarning)

EXCEL_PATH = "features_temperatures.xlsx"
SHEET = "caracteristicas_H"

EXCLUDE_COLS = [
    "id", "sample_idx", "tx_path", "rx_path", "status",
    "N_subcarriers", "M_symbols", "num_valid", "valid_ratio",
]

# --- Carga y limpieza ---
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET)
df = df[pd.to_numeric(df["temperature"], errors="coerce").notna()].copy()
df["temperature"] = df["temperature"].astype(float)

feature_cols = [
    c for c in df.columns
    if c not in EXCLUDE_COLS + ["temperature"]
    and c is not None
    and not str(c).startswith("Unnamed")
]
df = df[["temperature"] + feature_cols].dropna()

X = df[feature_cols].values
y = df["temperature"].values

# --- Split y escalado ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# --- Modelos ---
alphas_ridge = np.logspace(-2, 6, 200)
alphas_lasso = np.logspace(-3, 2, 100)

models = {
    "OLS":   LinearRegression(),
    "Ridge": RidgeCV(alphas=alphas_ridge, cv=5),
    "Lasso": LassoCV(alphas=alphas_lasso, cv=5, max_iter=100000, random_state=42),
}

results = {}
for name, m in models.items():
    m.fit(X_train_sc, y_train)
    y_pred = m.predict(X_test_sc)
    results[name] = {
        "model":  m,
        "y_pred": y_pred,
        "r2":     r2_score(y_test, y_pred),
        "rmse":   np.sqrt(mean_squared_error(y_test, y_pred)),
        "coef":   m.coef_,
    }

# --- Resumen de métricas ---
print(f"{'Modelo':<8}  {'R²':>8}  {'RMSE (°C)':>10}  {'Alpha':>12}  {'Features activas':>16}")
print("-" * 62)
for name, r in results.items():
    alpha_str = ""
    active = len(feature_cols)
    if hasattr(r["model"], "alpha_"):
        alpha_str = f"{r['model'].alpha_:.4g}"
    if name == "Lasso":
        active = int(np.sum(r["coef"] != 0))
    print(f"{name:<8}  {r['r2']:>8.4f}  {r['rmse']:>10.4f}  {alpha_str:>12}  {active:>16}")

# --- Coeficientes de Ridge y Lasso ---
for name in ("Ridge", "Lasso"):
    coef_df = pd.DataFrame({
        "feature":  feature_cols,
        "coef":     results[name]["coef"],
        "abs_coef": np.abs(results[name]["coef"]),
    }).sort_values("abs_coef", ascending=False)
    print(f"\nTop 15 coeficientes {name} (alpha={results[name]['model'].alpha_:.4g}):")
    print(coef_df.head(15).to_string(index=False))

if hasattr(results["Lasso"]["model"], "alpha_"):
    lasso_zero = int(np.sum(results["Lasso"]["coef"] == 0))
    print(f"\nLasso eliminó {lasso_zero}/{len(feature_cols)} features (coef = 0)")

# --- Figura ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

colors_model = {"OLS": "gray", "Ridge": "steelblue", "Lasso": "darkorange"}

for col, name in enumerate(["OLS", "Ridge", "Lasso"]):
    r = results[name]
    ax = axes[0, col]
    ax.scatter(y_test, r["y_pred"], alpha=0.6, color=colors_model[name],
               edgecolors="k", linewidths=0.3)
    lims = [min(y_test.min(), r["y_pred"].min()) - 1,
            max(y_test.max(), r["y_pred"].max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_xlabel("Temperatura real (°C)")
    ax.set_ylabel("Temperatura predicha (°C)")
    ax.set_title(f"{name}\nR²={r['r2']:.4f}  RMSE={r['rmse']:.4f} °C")

# Top 15 coeficientes Ridge
coef_ridge = pd.DataFrame({"feature": feature_cols, "coef": results["Ridge"]["coef"],
                            "abs_coef": np.abs(results["Ridge"]["coef"])}).sort_values("abs_coef", ascending=False)
top15r = coef_ridge.head(15)
colors_r = ["steelblue" if c > 0 else "tomato" for c in top15r["coef"]]
axes[1, 1].barh(top15r["feature"][::-1], top15r["coef"][::-1], color=colors_r[::-1])
axes[1, 1].axvline(0, color="k", linewidth=0.8)
axes[1, 1].set_title(f"Ridge — Top 15 coeficientes\n(alpha={results['Ridge']['model'].alpha_:.4g})")
axes[1, 1].set_xlabel("Coeficiente estandarizado")

# Top 15 coeficientes Lasso (solo no nulos)
coef_lasso = pd.DataFrame({"feature": feature_cols, "coef": results["Lasso"]["coef"],
                            "abs_coef": np.abs(results["Lasso"]["coef"])})
coef_lasso_nz = coef_lasso[coef_lasso["coef"] != 0].sort_values("abs_coef", ascending=False)
top15l = coef_lasso_nz.head(15)
colors_l = ["darkorange" if c > 0 else "tomato" for c in top15l["coef"]]
axes[1, 2].barh(top15l["feature"][::-1], top15l["coef"][::-1], color=colors_l[::-1])
axes[1, 2].axvline(0, color="k", linewidth=0.8)
axes[1, 2].set_title(f"Lasso — Top 15 coeficientes no nulos\n(alpha={results['Lasso']['model'].alpha_:.4g})")
axes[1, 2].set_xlabel("Coeficiente estandarizado")

# Comparativa de métricas
ax_comp = axes[1, 0]
names = list(results.keys())
r2s  = [results[n]["r2"]   for n in names]
rmses = [results[n]["rmse"] for n in names]
x = np.arange(len(names))
w = 0.35
bars1 = ax_comp.bar(x - w/2, r2s,  w, label="R²",        color=["gray","steelblue","darkorange"])
bars2 = ax_comp.bar(x + w/2, rmses, w, label="RMSE (°C)", color=["gray","steelblue","darkorange"], alpha=0.5, hatch="//")
ax_comp.set_xticks(x)
ax_comp.set_xticklabels(names)
ax_comp.set_title("Comparativa OLS vs Ridge vs Lasso")
ax_comp.legend()
for bar in bars1:
    ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("regresion_temperatura.png", dpi=150)
plt.show()
print("\nFigura guardada en regresion_temperatura.png")
