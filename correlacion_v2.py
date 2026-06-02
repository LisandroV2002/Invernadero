"""
Análisis de correlación T_exterior vs T_interior
=================================================
Paso 1 → Gráficas de diagnóstico para justificar el corte horario
Paso 2 → Correlación por segmento:
    · Divergencia : 10 h – 20 h  (T_interior se dispara sobre T_exterior)
    · Equilibrio  : 21 h –  9 h  (temperaturas siguen juntas)

Uso: python3 analisis_correlacion_segmentada.py datos_combinados.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# 0 · CONFIG
# ══════════════════════════════════════════════════════════════════
CSV_PATH   = sys.argv[1] if len(sys.argv) > 1 else "datos_combinados.csv"

H_DIV_INI  = 10   # hora inicio divergencia
H_DIV_FIN  = 20   # hora fin   divergencia

# Semana a mostrar en el time-series (días completos en el dataset)
SEMANA_INI = "2026-05-08"
SEMANA_FIN = "2026-05-13"

COLORS = {
    "T_int"   : "#D85A30",   # coral
    "T_ext"   : "#378ADD",   # azul
    "div"     : "#D85A30",   # divergencia
    "equ"     : "#378ADD",   # equilibrio
    "div_bg"  : "#FAECE7",   # fondo zona divergencia
    "equ_bg"  : "#E6F1FB",   # fondo zona equilibrio
    "grid"    : "#DDDDDD",
}

# ══════════════════════════════════════════════════════════════════
# 1 · CARGA Y PREPARACIÓN
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv(CSV_PATH)
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
df["hora"]       = df["fecha_hora"].dt.hour
df["fecha"]      = df["fecha_hora"].dt.date
df               = df.dropna(subset=["T_interior", "T_exterior"])

# Etiqueta de segmento
df["segmento"] = df["hora"].apply(
    lambda h: "Divergencia" if H_DIV_INI <= h <= H_DIV_FIN else "Equilibrio"
)

print(f"\n{'='*58}")
print(f"  Archivo  : {CSV_PATH}")
print(f"  Registros: {len(df)}")
print(f"  Período  : {df['fecha_hora'].min()} → {df['fecha_hora'].max()}")
print(f"{'='*58}\n")

# ══════════════════════════════════════════════════════════════════
# 2 · ESTADÍSTICAS POR HORA (para todas las gráficas)
# ══════════════════════════════════════════════════════════════════
por_hora = (
    df.groupby("hora")
    .agg(
        delta_mean  = ("delta_T", "mean"),
        delta_std   = ("delta_T", "std"),
        delta_p25   = ("delta_T", lambda x: x.quantile(0.25)),
        delta_p75   = ("delta_T", lambda x: x.quantile(0.75)),
        delta_p10   = ("delta_T", lambda x: x.quantile(0.10)),
        delta_p90   = ("delta_T", lambda x: x.quantile(0.90)),
        Tint_mean   = ("T_interior", "mean"),
        Text_mean   = ("T_exterior", "mean"),
        n           = ("delta_T", "count"),
    )
    .reset_index()
)

# ══════════════════════════════════════════════════════════════════
# 3 · GRÁFICA 1 — Serie temporal (semana completa)
# ══════════════════════════════════════════════════════════════════
semana = df[
    (df["fecha_hora"] >= SEMANA_INI) &
    (df["fecha_hora"] <= SEMANA_FIN + " 23:59")
].copy()

fig1, ax = plt.subplots(figsize=(14, 5))
fig1.patch.set_facecolor("white")

# Sombrear franjas de divergencia
for dia in semana["fecha"].unique():
    base_ini = pd.Timestamp(str(dia)) + pd.Timedelta(hours=H_DIV_INI)
    base_fin = pd.Timestamp(str(dia)) + pd.Timedelta(hours=H_DIV_FIN + 1)
    ax.axvspan(base_ini, base_fin, color=COLORS["div_bg"], alpha=0.9, zorder=0)

ax.plot(semana["fecha_hora"], semana["T_interior"],
        color=COLORS["T_int"], lw=2,   label="T interior", zorder=3)
ax.plot(semana["fecha_hora"], semana["T_exterior"],
        color=COLORS["T_ext"], lw=2,   label="T exterior", zorder=3)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
ax.set_ylabel("Temperatura (°C)", fontsize=11)
ax.set_title("T interior vs T exterior — semana representativa\n"
             "(zonas sombreadas = horas de divergencia 10 – 20 h)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, color=COLORS["grid"], linewidth=0.6)
ax.set_facecolor("white")

patch_div = mpatches.Patch(color=COLORS["div_bg"], label="Zona divergencia (10–20 h)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [patch_div], labels + ["Zona divergencia (10–20 h)"],
          fontsize=10, loc="upper left")

plt.tight_layout()
fig1.savefig("grafica1_serie_temporal.png", dpi=150, bbox_inches="tight")
print("  → grafica1_serie_temporal.png")

# ══════════════════════════════════════════════════════════════════
# 4 · GRÁFICA 2 — T_int y T_ext promedio por hora (perfil diario)
# ══════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(12, 5))
fig2.patch.set_facecolor("white")

ax.axvspan(H_DIV_INI - 0.5, H_DIV_FIN + 0.5,
           color=COLORS["div_bg"], alpha=0.9, zorder=0,
           label="Zona divergencia (10–20 h)")

ax.plot(por_hora["hora"], por_hora["Tint_mean"],
        "o-", color=COLORS["T_int"], lw=2.5, ms=6, label="T interior media")
ax.plot(por_hora["hora"], por_hora["Text_mean"],
        "s--", color=COLORS["T_ext"], lw=2.5, ms=6, label="T exterior media")

ax.fill_between(
    por_hora["hora"],
    por_hora["Tint_mean"],
    por_hora["Text_mean"],
    where=(por_hora["hora"] >= H_DIV_INI) & (por_hora["hora"] <= H_DIV_FIN),
    alpha=0.25, color=COLORS["T_int"], label="Diferencia (divergencia)"
)

ax.set_xticks(range(24))
ax.set_xlabel("Hora del día", fontsize=11)
ax.set_ylabel("Temperatura media (°C)", fontsize=11)
ax.set_title("Perfil diario promedio: T interior vs T exterior\n"
             "La brecha visible confirma el rango de divergencia",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(fontsize=10)
ax.grid(True, color=COLORS["grid"], linewidth=0.6)
ax.set_facecolor("white")

plt.tight_layout()
fig2.savefig("grafica2_perfil_diario.png", dpi=150, bbox_inches="tight")
print("  → grafica2_perfil_diario.png")


# ══════════════════════════════════════════════════════════════════
# 7 · GRÁFICA 5 — Scatter T_ext vs T_int coloreado por segmento
# ══════════════════════════════════════════════════════════════════
fig5, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig5.patch.set_facecolor("white")

for i, (seg, col) in enumerate([("Divergencia", COLORS["div"]),
                                  ("Equilibrio",   COLORS["equ"])]):
    ax_s = axes[i]
    sub  = df[df["segmento"] == seg]

    ax_s.scatter(sub["T_exterior"], sub["T_interior"],
                 c=col, alpha=0.45, s=18, zorder=3)

    # Línea de regresión
    slope, intercept, r, p, _ = stats.linregress(sub["T_exterior"], sub["T_interior"])
    x_range = np.linspace(sub["T_exterior"].min(), sub["T_exterior"].max(), 100)
    ax_s.plot(x_range, slope * x_range + intercept,
              color="#222222", lw=1.8, linestyle="--", zorder=4,
              label=f"y = {slope:.2f}x + {intercept:.1f}")

    # Línea identidad (T_int = T_ext)
    lims = [
        min(sub["T_exterior"].min(), sub["T_interior"].min()) - 1,
        max(sub["T_exterior"].max(), sub["T_interior"].max()) + 1,
    ]
    ax_s.plot(lims, lims, color="#888888", lw=1.2, linestyle=":",
              zorder=2, label="T_int = T_ext")
    ax_s.set_xlim(lims); ax_s.set_ylim(lims)

    ax_s.set_title(f"{seg} ({H_DIV_INI}–{H_DIV_FIN} h)"
                   if seg == "Divergencia"
                   else f"{seg} (21–9 h)",
                   fontsize=12, fontweight="bold")
    ax_s.set_xlabel("T exterior (°C)", fontsize=11)
    if i == 0:
        ax_s.set_ylabel("T interior (°C)", fontsize=11)
    ax_s.legend(fontsize=9)
    ax_s.grid(True, color=COLORS["grid"], linewidth=0.6)
    ax_s.set_facecolor("white")

    r2 = r**2
    n  = len(sub)
    ax_s.text(0.04, 0.96,
              f"r = {r:+.4f}\nR² = {r2:.4f}\nN = {n}",
              transform=ax_s.transAxes, fontsize=11, va="top",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        edgecolor="#cccccc", alpha=0.85))

fig5.suptitle("Scatter T_exterior vs T_interior por segmento\n"
              "La línea punteada gris es T_int = T_ext (referencia perfecta)",
              fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig5.savefig("grafica5_scatter_segmentos.png", dpi=150, bbox_inches="tight")
print("  → grafica5_scatter_segmentos.png")

# ══════════════════════════════════════════════════════════════════
# 8 · CORRELACIÓN FINAL
# ══════════════════════════════════════════════════════════════════
def correlacion(subset, etiqueta):
    x = subset["T_exterior"].values
    y = subset["T_interior"].values
    r, p = stats.pearsonr(x, y)
    r2   = r ** 2
    n    = len(x)
    sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {etiqueta:<30}  r = {r:+.4f}   R² = {r2:.4f}   "
          f"p = {p:.4e} {sig}   N = {n}")
    return {"Segmento": etiqueta, "Pearson_r": r, "R2": r2,
            "p_value": p, "Significancia": sig, "N": n}

print(f"\n{'─'*60}")
print("  CORRELACIÓN FINAL POR SEGMENTO")
print(f"  Divergencia: {H_DIV_INI}:00 – {H_DIV_FIN}:00 h"
      f"  |  Equilibrio: 21:00 – 09:00 h")
print(f"{'─'*60}")

resultados = [
    correlacion(df[df["segmento"] == "Divergencia"], f"Divergencia ({H_DIV_INI}–{H_DIV_FIN} h)"),
    correlacion(df[df["segmento"] == "Equilibrio"],  "Equilibrio  (21–9 h)"),
]

resumen = pd.DataFrame(resultados)
resumen.to_csv("resultados_correlacion_segmentada.csv", index=False)

print(f"\n  Significancia: *** p<0.001 | ** p<0.01 | * p<0.05 | ns no significativo")
print(f"  Tabla guardada en: resultados_correlacion_segmentada.csv")
print(f"{'='*60}\n")

plt.show()
print("  Todas las gráficas generadas correctamente.")
