import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning

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

# Matriz de correlación (incluye temperatura como primera variable)
all_cols = ["temperature"] + feature_cols
corr = df[all_cols].corr()

# --- Top correlaciones con temperatura ---
corr_temp = corr["temperature"].drop("temperature").sort_values(key=abs, ascending=False)
print("Top 20 correlaciones con temperatura (Pearson):")
print(corr_temp.head(20).to_string())

# --- Todos los pares de features con alta colinealidad ---
corr_feat = corr.loc[feature_cols, feature_cols]
mask_upper = np.triu(np.ones(corr_feat.shape, dtype=bool), k=1)
pairs = (
    corr_feat.where(mask_upper)
    .stack()
    .reset_index()
    .rename(columns={"level_0": "var1", "level_1": "var2", 0: "r"})
    .assign(abs_r=lambda x: x["r"].abs())
    .sort_values("abs_r", ascending=False)
)

def banda(r):
    a = abs(r)
    if a == 1.0:   return "r = ±1.00 (idénticas)"
    if a >= 0.99:  return "0.99 <= |r| < 1.00"
    if a >= 0.95:  return "0.95 <= |r| < 0.99"
    return               "0.90 <= |r| < 0.95"

pairs_high = pairs[pairs["abs_r"] >= 0.90].copy()
pairs_high["banda"] = pairs_high["r"].apply(banda)

print(f"\nPares de features con |r| >= 0.90: {len(pairs_high)}\n")
for band_label in ["r = ±1.00 (idénticas)", "0.99 <= |r| < 1.00",
                   "0.95 <= |r| < 0.99",     "0.90 <= |r| < 0.95"]:
    sub = pairs_high[pairs_high["banda"] == band_label][["var1", "var2", "r"]]
    print(f"── {band_label} ({len(sub)} pares) ──")
    print(sub.to_string(index=False))
    print()

# Guardar tabla completa en Excel
pairs_high[["var1", "var2", "r", "banda"]].to_excel(
    "colinealidad_features.xlsx", index=False, sheet_name="pares_correlados"
)
print("Tabla guardada en colinealidad_features.xlsx")

# ── Figura 1: heatmap completo con clustering ──────────────────────────────
fig1, ax1 = plt.subplots(figsize=(18, 15))
sns.heatmap(
    corr,
    ax=ax1,
    cmap="RdBu_r",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0,
    cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    xticklabels=True,
    yticklabels=True,
)
ax1.set_title("Matriz de correlaciones cruzadas (todas las variables)", fontsize=13, pad=12)
ax1.tick_params(axis="x", labelsize=6, rotation=90)
ax1.tick_params(axis="y", labelsize=6, rotation=0)
plt.tight_layout()
plt.savefig("correlaciones_heatmap.png", dpi=150)
print("\nFigura 1 guardada en correlaciones_heatmap.png")

# ── Figura 2: clustermap (reordena por similitud) ──────────────────────────
# NaN en la matriz (features perfectamente correlacionadas) se sustituyen por 0
# solo para el cálculo del dendrograma; los valores originales se muestran en el mapa
corr_cluster = corr.fillna(0)
g = sns.clustermap(
    corr_cluster,
    cmap="RdBu_r",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0,
    figsize=(20, 17),
    cbar_pos=(0.02, 0.8, 0.03, 0.15),
    dendrogram_ratio=0.12,
    xticklabels=True,
    yticklabels=True,
)
g.ax_heatmap.tick_params(axis="x", labelsize=5.5, rotation=90)
g.ax_heatmap.tick_params(axis="y", labelsize=5.5, rotation=0)
g.figure.suptitle("Clustermap de correlaciones (variables agrupadas por similitud)", y=1.01, fontsize=13)
g.figure.savefig("correlaciones_clustermap.png", dpi=150, bbox_inches="tight")
print("Figura 2 guardada en correlaciones_clustermap.png")

# ── Figura 3: correlación de cada feature con temperatura ─────────────────
fig3, ax3 = plt.subplots(figsize=(10, 14))
colors = ["steelblue" if v > 0 else "tomato" for v in corr_temp.values]
ax3.barh(corr_temp.index[::-1], corr_temp.values[::-1], color=colors[::-1])
ax3.axvline(0, color="k", linewidth=0.8)
ax3.axvline( 0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
ax3.axvline(-0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
ax3.set_xlabel("Correlación de Pearson con temperatura")
ax3.set_title("Correlación de cada feature con la temperatura")
ax3.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.savefig("correlaciones_con_temperatura.png", dpi=150)
print("Figura 3 guardada en correlaciones_con_temperatura.png")

plt.show()
