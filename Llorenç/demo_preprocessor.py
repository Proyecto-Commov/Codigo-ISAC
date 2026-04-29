"""
demo_preprocessor.py
====================
Demostración del módulo preprocessor.py:
  - Espectrogramas antes/después del filtrado
  - Análisis de tasa de falsas alarmas con/sin filtro
  - Comparativa de métodos de clutter y separación

Genera: demo_output/  con todas las figuras en PNG.
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import stft, get_window

# Añadimos el directorio donde está preprocessor.py
sys.path.insert(0, os.path.dirname(__file__))
from preprocessor import (
    PreprocessConfig, ClutterRemover, MicroDopplerSeparator, Preprocessor
)

os.makedirs("demo_output", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Parámetros globales de simulación
# ─────────────────────────────────────────────────────────────────────────────
FS_SLOW = 200.0      # Hz (tasa de símbolos OFDM = 1/T_símbolo)
DURATION = 4.0       # segundos
N = int(FS_SLOW * DURATION)
t = np.linspace(0, DURATION, N, endpoint=False)
RNG = np.random.default_rng(0)

# Config adaptada al FS_SLOW sintético
def make_cfg(**kwargs) -> PreprocessConfig:
    cfg = PreprocessConfig(
        sample_rate=FS_SLOW * (1024 + 128),
        n_subcarriers=1024,
        cp_len=128,
        human_doppler_low_hz=0.5,
        human_doppler_high_hz=min(20.0, FS_SLOW / 2 - 1),
        human_energy_ratio_threshold=0.15,
        nperseg=min(64, N // 8),
        noverlap=min(48, N // 8 - 1),
        nfft=128,
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg

# ─────────────────────────────────────────────────────────────────────────────
# Generadores de señal sintética
# ─────────────────────────────────────────────────────────────────────────────

def make_raw_signal(include_human=True, include_thermal=True,
                    clutter_amp=5.0, human_amp=0.5, thermal_amp=0.15,
                    noise_amp=0.05, n_series=6) -> np.ndarray:
    """
    Compone una señal realista:
      - Clutter estático (constante)
      - Componente humana: torso ~1.5 Hz + piernas ~6 Hz + brazos ~4 Hz
      - Componente térmica: deriva muy lenta ~0.08 Hz (convección)
      - Ruido blanco gaussiano
    """
    x = clutter_amp * (1.0 + 0.3j) * np.ones((n_series, N))

    if include_human:
        h  = human_amp * np.exp(1j * 2 * np.pi * 1.5 * t)      # torso
        h += (human_amp * 0.6) * np.exp(1j * 2 * np.pi * 6.0 * t)   # piernas
        h += (human_amp * 0.4) * np.exp(1j * 2 * np.pi * 4.0 * t)   # brazos
        h += (human_amp * 0.3) * np.exp(1j * 2 * np.pi * 0.3 * t)   # respiración
        x += h[np.newaxis, :]

    if include_thermal:
        th  = thermal_amp * np.exp(1j * 2 * np.pi * 0.08 * t)
        th += thermal_amp * 0.5 * RNG.standard_normal(N)        # ruido coloreado
        x += th[np.newaxis, :]

    x += noise_amp * (
        RNG.standard_normal((n_series, N)) +
        1j * RNG.standard_normal((n_series, N))
    )
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de visualización
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectrogram(series_1d: np.ndarray, fs: float, nperseg=64,
                        noverlap=48, nfft=128) -> tuple:
    win = get_window("hann", nperseg)
    f, t_stft, Zxx = stft(
        series_1d, fs=fs, window=win, nperseg=nperseg,
        noverlap=noverlap, nfft=nfft,
        return_onesided=False, boundary=None, padded=False
    )
    f_s = np.fft.fftshift(f)
    S   = np.abs(np.fft.fftshift(Zxx, axes=0)) ** 2
    return f_s, t_stft, 10 * np.log10(S + 1e-30)


def plot_spectrogram_ax(ax, f, t, S_db, title, vmin=None, vmax=None,
                        cmap="inferno"):
    im = ax.pcolormesh(t, f, S_db, shading="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)
    ax.set_xlabel("Tiempo [s]", fontsize=9)
    ax.set_ylabel("Frecuencia Doppler [Hz]", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(-FS_SLOW / 2, FS_SLOW / 2)
    return im


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 1: Antes y después del filtro de clutter (3 métodos)
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Figura 1: Clutter removal comparativa…")

x_raw = make_raw_signal(n_series=6)
serie_idx = 2   # seleccionamos la serie 2 para visualizar
serie_raw = x_raw[serie_idx].real

methods_clutter = ["dc_sub", "ema", "svd"]
titles_clutter  = ["Sustracción DC", "EMA High-pass (α=0.98)", "SVD rank-1"]

fig, axes = plt.subplots(2, 4, figsize=(18, 7))
fig.suptitle("Espectrogramas: Antes y Después de la Eliminación de Clutter\n"
             "(Señal = Clutter estático + Humano + Térmico + Ruido)",
             fontsize=12, fontweight="bold")

# Espectrograma de la señal cruda (panel izquierdo)
f0, t0, S0 = compute_spectrogram(serie_raw, FS_SLOW)
vmin_ref, vmax_ref = np.percentile(S0, [5, 99])

im = plot_spectrogram_ax(axes[0, 0], f0, t0, S0,
                         "CRUDO (con clutter)", vmin=vmin_ref, vmax=vmax_ref)
axes[0, 0].set_ylim(-25, 25)   # zoom para ver la dinámica
plt.colorbar(im, ax=axes[0, 0], label="dB")

# Espectrograma de la parte thermal (referencia)
x_thermal_only = make_raw_signal(include_human=False, n_series=6)
f_th, t_th, S_th = compute_spectrogram(x_thermal_only[serie_idx].real, FS_SLOW)
im2 = plot_spectrogram_ax(axes[1, 0], f_th, t_th, S_th,
                          "Solo térmico (referencia)", vmin=vmin_ref, vmax=vmax_ref)
axes[1, 0].set_ylim(-25, 25)
plt.colorbar(im2, ax=axes[1, 0], label="dB")

for col, (meth, title) in enumerate(zip(methods_clutter, titles_clutter), start=1):
    cfg = make_cfg(clutter_method=meth)
    cr  = ClutterRemover(cfg)
    y, info = cr.remove(x_raw)
    f_c, t_c, S_c = compute_spectrogram(y[serie_idx].real, FS_SLOW)
    vmin_c, vmax_c = np.percentile(S_c, [5, 99])

    # Fila superior: espectrograma filtrado
    im_top = plot_spectrogram_ax(axes[0, col], f_c, t_c, S_c,
                                 f"Filtrado: {title}", vmin=vmin_c, vmax=vmax_c)
    axes[0, col].set_ylim(-25, 25)
    plt.colorbar(im_top, ax=axes[0, col], label="dB")

    # Fila inferior: diferencia crudo - filtrado (clutter eliminado)
    f_d, t_d, S_diff = compute_spectrogram(
        (x_raw[serie_idx] - y[serie_idx]).real, FS_SLOW
    )
    im_bot = plot_spectrogram_ax(axes[1, col], f_d, t_d, S_diff,
                                 f"Clutter extraído ({title})",
                                 cmap="viridis")
    axes[1, col].set_ylim(-25, 25)
    plt.colorbar(im_bot, ax=axes[1, col], label="dB")

    # Anotación con métrica de supresión
    if meth == "dc_sub":
        sup_db = info["dc_sub"].get("suppression_dB", 0)
        axes[0, col].text(0.02, 0.97, f"Sup: {sup_db:.1f} dB",
                         transform=axes[0, col].transAxes,
                         color="white", fontsize=8, va="top",
                         bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
    elif meth == "svd":
        frac = info["svd"]["energy_fraction_removed"]
        axes[0, col].text(0.02, 0.97, f"Energía removida: {frac:.0%}",
                         transform=axes[0, col].transAxes,
                         color="white", fontsize=8, va="top",
                         bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

plt.tight_layout()
plt.savefig("demo_output/fig1_clutter_removal.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → demo_output/fig1_clutter_removal.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 2: Separación humano vs. térmico
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Figura 2: Separación de fuentes…")

x_mixed = make_raw_signal(n_series=6, human_amp=0.6, thermal_amp=0.2)

cfg_sep = make_cfg(clutter_method="dc_sub", separation_method="combined")
prep    = Preprocessor(cfg_sep)
result  = prep.run(x_mixed)

fig2, axes2 = plt.subplots(3, 3, figsize=(16, 12))
fig2.suptitle("Separación Micro-Doppler: Señal Mixta → Componente Humana + Térmica\n"
              "(Método: combined  |  Clutter: DC subtraction)",
              fontsize=12, fontweight="bold")

# Tres series representativas
for col, s_idx in enumerate([0, 2, 4]):
    def spect(arr):
        return compute_spectrogram(arr[s_idx].real, FS_SLOW)

    titles_row = [
        f"Serie {s_idx}: Tras eliminar clutter",
        f"Serie {s_idx}: Componente HUMANA",
        f"Serie {s_idx}: Componente TÉRMICA",
    ]
    arrays = [result.after_clutter, result.human_component, result.thermal_component]
    cmaps  = ["inferno", "hot", "cool"]

    for row, (arr, title, cmap) in enumerate(zip(arrays, titles_row, cmaps)):
        f_, t_, S_ = spect(arr)
        vmin_, vmax_ = np.percentile(S_, [5, 99])
        im_ = plot_spectrogram_ax(axes2[row, col], f_, t_, S_,
                                  title, vmin=vmin_, vmax=vmax_, cmap=cmap)
        axes2[row, col].set_ylim(-25, 25)
        plt.colorbar(im_, ax=axes2[row, col], label="dB")

        # Indicador de clasificación
        is_human = result.human_mask[s_idx]
        if row == 0:
            label = "[H] HUMANO" if is_human else "[T] TERMICO"
            color = "#00e676" if is_human else "#ff7043"
            axes2[row, col].text(
                0.98, 0.97, label,
                transform=axes2[row, col].transAxes,
                color=color, fontsize=9, va="top", ha="right", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7)
            )

plt.tight_layout()
plt.savefig("demo_output/fig2_source_separation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → demo_output/fig2_source_separation.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 3: Análisis de tasa de falsas alarmas (FA)
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Figura 3: Análisis de tasa de falsas alarmas…")

thresholds = np.linspace(0.02, 0.70, 30)
n_montecarlo = 40   # series por experimento

# --- Señales puramente térmicas (sin humano) → FA si clasificado como humano
x_thermal_fa = make_raw_signal(include_human=False, include_thermal=True,
                                thermal_amp=0.3, noise_amp=0.05,
                                n_series=n_montecarlo)

# --- Señales puramente humanas → Miss si NO clasificado como humano
x_human_only = make_raw_signal(include_human=True, include_thermal=False,
                                human_amp=0.5, noise_amp=0.05,
                                n_series=n_montecarlo)

fa_rates_no_filter   = []   # sin ningún filtro (clasificación directa)
fa_rates_with_filter = []   # con eliminación de clutter previa
miss_rates           = []
miss_rates_no_filter = []

for thr in thresholds:
    # Sin filtro de clutter
    cfg_no = make_cfg(clutter_method="dc_sub",
                      separation_method="bandpass",
                      human_energy_ratio_threshold=thr)
    sep_no = MicroDopplerSeparator(cfg_no)
    _, _, mask_fa_no, _  = sep_no.separate(x_thermal_fa)
    _, _, mask_miss_no, _ = sep_no.separate(x_human_only)
    fa_rates_no_filter.append(mask_fa_no.mean())
    miss_rates_no_filter.append(1.0 - mask_miss_no.mean())

    # Con filtro de clutter (SVD)
    cfg_f = make_cfg(clutter_method="svd",
                     separation_method="bandpass",
                     svd_rank=1,
                     human_energy_ratio_threshold=thr)
    cr_f  = ClutterRemover(cfg_f)
    x_th_clean, _  = cr_f.remove(x_thermal_fa)
    x_hu_clean, _  = cr_f.remove(x_human_only)
    sep_f = MicroDopplerSeparator(cfg_f)
    _, _, mask_fa_f, _   = sep_f.separate(x_th_clean)
    _, _, mask_miss_f, _ = sep_f.separate(x_hu_clean)
    fa_rates_with_filter.append(mask_fa_f.mean())
    miss_rates.append(1.0 - mask_miss_f.mean())

fa_no  = np.array(fa_rates_no_filter)
fa_f   = np.array(fa_rates_with_filter)
miss_n = np.array(miss_rates_no_filter)
miss_f = np.array(miss_rates)

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle("Análisis de Tasa de Falsas Alarmas (FA) y Pérdidas de Detección",
              fontsize=12, fontweight="bold")

# Panel 1: FA rate vs umbral
ax = axes3[0]
ax.plot(thresholds, fa_no * 100, "r--o", ms=4, label="Sin filtro clutter")
ax.plot(thresholds, fa_f * 100,  "b-o",  ms=4, label="Con filtro SVD")
ax.set_xlabel("Umbral de ratio energético", fontsize=10)
ax.set_ylabel("Tasa de FA [%]", fontsize=10)
ax.set_title("FA Rate vs Umbral\n(señal: solo térmica → FA si detecta humano)")
ax.legend(); ax.grid(alpha=0.3)
ax.set_ylim(-5, 105)

# Panel 2: Miss rate vs umbral
ax2 = axes3[1]
ax2.plot(thresholds, miss_n * 100, "r--o", ms=4, label="Sin filtro clutter")
ax2.plot(thresholds, miss_f * 100, "b-o",  ms=4, label="Con filtro SVD")
ax2.set_xlabel("Umbral de ratio energético", fontsize=10)
ax2.set_ylabel("Tasa de pérdidas [%]", fontsize=10)
ax2.set_title("Miss Rate vs Umbral\n(señal: solo humana → miss si NO detecta)")
ax2.legend(); ax2.grid(alpha=0.3)
ax2.set_ylim(-5, 105)

# Panel 3: Curva ROC (FA vs 1-miss = Pd)
ax3 = axes3[2]
ax3.plot(fa_no * 100, (1 - miss_n) * 100, "r--o", ms=4, label="Sin filtro clutter")
ax3.plot(fa_f * 100,  (1 - miss_f) * 100, "b-o",  ms=4, label="Con filtro SVD")
ax3.plot([0, 100], [0, 100], "k:", alpha=0.4, label="Clasificador aleatorio")
ax3.set_xlabel("FA Rate [%]", fontsize=10)
ax3.set_ylabel("Probabilidad de detección Pd [%]", fontsize=10)
ax3.set_title("Curva ROC\n(FA Rate vs Pd)")
ax3.legend(); ax3.grid(alpha=0.3)
ax3.set_xlim(-2, 102); ax3.set_ylim(-2, 102)

# AUC aproximada bajo la ROC
for lbl, fa_arr, miss_arr in [
    ("Sin filtro", fa_no, miss_n),
    ("Con filtro", fa_f,  miss_f),
]:
    pd_arr = (1 - miss_arr)
    sort_idx = np.argsort(fa_arr)
    auc = np.trapz(pd_arr[sort_idx], fa_arr[sort_idx])
    ax3.text(0.02, 0.05 + (0.12 if "Sin" in lbl else 0.0),
             f"AUC {lbl}: {auc:.3f}",
             transform=ax3.transAxes, fontsize=8,
             bbox=dict(boxstyle="round,pad=0.2",
                       fc="lightcoral" if "Sin" in lbl else "lightblue", alpha=0.8))

plt.tight_layout()
plt.savefig("demo_output/fig3_false_alarm_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → demo_output/fig3_false_alarm_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 4: Pipeline completo antes/después para escenarios distintos
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Figura 4: Escenarios antes/después…")

scenarios = [
    ("Habitación vacía\n(solo clutter+ruido)",
     make_raw_signal(include_human=False, include_thermal=False,
                     clutter_amp=5, noise_amp=0.1, n_series=4)),
    ("Fuego detectado\n(clutter+térmico)",
     make_raw_signal(include_human=False, include_thermal=True,
                     clutter_amp=5, thermal_amp=0.4, noise_amp=0.05, n_series=4)),
    ("Persona en habitación\n(clutter+humano)",
     make_raw_signal(include_human=True, include_thermal=False,
                     clutter_amp=5, human_amp=0.7, noise_amp=0.05, n_series=4)),
    ("Persona + fuego\n(escenario complejo)",
     make_raw_signal(include_human=True, include_thermal=True,
                     clutter_amp=5, human_amp=0.6, thermal_amp=0.3,
                     noise_amp=0.05, n_series=4)),
]

cfg_full = make_cfg(clutter_method="svd", separation_method="combined")
prep_full = Preprocessor(cfg_full)

fig4, axes4 = plt.subplots(3, len(scenarios), figsize=(18, 10))
fig4.suptitle("Pipeline Completo: 4 Escenarios\n"
              "Fila 1: Crudo  |  Fila 2: Tras clutter SVD  |  Fila 3: Componente térmica",
              fontsize=11, fontweight="bold")

row_titles = ["Señal cruda", "Tras eliminar clutter", "Componente térmica extraída"]

for col, (sc_title, x_sc) in enumerate(scenarios):
    result_sc = prep_full.run(x_sc)
    arrays_sc = [x_sc, result_sc.after_clutter, result_sc.thermal_component]
    cmaps_sc  = ["inferno", "inferno", "cool"]

    for row, (arr_sc, cmap_sc) in enumerate(zip(arrays_sc, cmaps_sc)):
        f_, t_, S_ = compute_spectrogram(arr_sc[0].real, FS_SLOW)
        vmin_, vmax_ = np.percentile(S_, [5, 99])
        im_ = plot_spectrogram_ax(axes4[row, col], f_, t_, S_,
                                  sc_title if row == 0 else row_titles[row],
                                  vmin=vmin_, vmax=vmax_, cmap=cmap_sc)
        axes4[row, col].set_ylim(-20, 20)
        plt.colorbar(im_, ax=axes4[row, col], label="dB")

    # Etiqueta de clasificación
    n_hum = result_sc.human_mask.sum()
    n_total = len(result_sc.human_mask)
    txt = f"Humano: {n_hum}/{n_total} series"
    col_txt = "#00e676" if n_hum > 0 else "#ff7043"
    axes4[2, col].text(0.5, -0.22, txt, transform=axes4[2, col].transAxes,
                       ha="center", fontsize=9, color=col_txt, fontweight="bold")

plt.tight_layout()
plt.savefig("demo_output/fig4_scenarios.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → demo_output/fig4_scenarios.png")


# ─────────────────────────────────────────────────────────────────────────────
# Resumen en consola
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" RESUMEN DE RESULTADOS")
print("="*60)

# Calcular FA a umbral 0.15
thr_ref = 0.15
cfg_ref = make_cfg(clutter_method="svd", separation_method="bandpass",
                   human_energy_ratio_threshold=thr_ref)
cr_ref  = ClutterRemover(cfg_ref)
sep_ref = MicroDopplerSeparator(cfg_ref)

x_th_clean, _ = cr_ref.remove(
    make_raw_signal(include_human=False, include_thermal=True,
                    thermal_amp=0.3, n_series=50))
x_hu_clean, _ = cr_ref.remove(
    make_raw_signal(include_human=True, include_thermal=False,
                    human_amp=0.5, n_series=50))
_, _, mask_th, _ = sep_ref.separate(x_th_clean)
_, _, mask_hu, _ = sep_ref.separate(x_hu_clean)

fa_pct  = mask_th.mean() * 100
pd_pct  = mask_hu.mean() * 100
miss_pct = (1 - mask_hu.mean()) * 100

print(f"\n  Umbral de energía : {thr_ref}")
print(f"  Tasa FA           : {fa_pct:.1f}%  (señal térmica clasificada como humana)")
print(f"  Pd (detección)    : {pd_pct:.1f}%  (señal humana correctamente detectada)")
print(f"  Miss rate         : {miss_pct:.1f}%  (señal humana no detectada)")
print(f"\n  Figuras generadas en: demo_output/")
print("  fig1_clutter_removal.png")
print("  fig2_source_separation.png")
print("  fig3_false_alarm_analysis.png")
print("  fig4_scenarios.png")
print("="*60)
