"""
Correlación Invernadero — Radiación vs ΔT (día) y T_ext vs T_int (noche)
=========================================================================
Requiere:
    pip install openmeteo-requests requests-cache retry-requests pandas scipy matplotlib
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import openmeteo_requests
import requests_cache
from retry_requests import retry


# ════════════════════════════════════════════════════════════════════════════
# 1. CARGAR TEMPERATURA INTERIOR  (datos_interior.csv — formato largo)
# ════════════════════════════════════════════════════════════════════════════
df_int_raw = pd.read_csv("datos_interior.csv", parse_dates=["fecha_hora"])

# Nos quedamos sólo con el sensor de temperatura ambiente a 2 m
df_int = (
    df_int_raw[df_int_raw["nombre_sensor"] == "TEMP_AMB_2M"]
    [["fecha_hora", "valor_medicion"]]
    .rename(columns={"valor_medicion": "T_interior"})
    .copy()
)

df_int["fecha_hora"] = pd.to_datetime(df_int["fecha_hora"]).dt.tz_localize(None)


# ════════════════════════════════════════════════════════════════════════════
# 2. CARGAR TEMPERATURA EXTERIOR  (temperatura_exterior.csv — formato ancho)
# ════════════════════════════════════════════════════════════════════════════
df_ext = pd.read_csv("temperatura_exterior.csv", parse_dates=["fecha_hora"])
df_ext = df_ext.rename(columns={"temperatura_85": "T_exterior"})   # typo en el CSV original

# La columna no tiene tz → la tratamos directamente como UTC
df_ext["fecha_hora"] = pd.to_datetime(df_ext["fecha_hora"])


# ════════════════════════════════════════════════════════════════════════════
# 3. OBTENER RADIACIÓN DESDE OPEN-METEO
# ════════════════════════════════════════════════════════════════════════════

df_rad = pd.read_csv("open-meteo-33.36S66.22W826m.csv", skiprows=3)
df_rad.columns = ['fecha_hora', 'shortwave_radiation']
df_rad['fecha_hora'] = pd.to_datetime(df_rad['fecha_hora'])

# ════════════════════════════════════════════════════════════════════════════
# 4. MERGE DE LOS TRES DATAFRAMES (join por fecha_hora exacta)
# ════════════════════════════════════════════════════════════════════════════
df = (
    df_int
    .merge(df_ext, on="fecha_hora", how="inner")
    .merge(df_rad, on="fecha_hora", how="inner")
    .dropna(subset=["T_interior", "T_exterior", "shortwave_radiation"])
    .sort_values("fecha_hora")
    .reset_index(drop=True)
)

df["delta_T"] = df["T_interior"] - df["T_exterior"]

print(f"\nRegistros tras el merge: {len(df)}")
print(df[["fecha_hora", "T_interior", "T_exterior", "delta_T", "shortwave_radiation"]].head(10))

# ── Exportar el DataFrame combinado a CSV ────────────────────────────────────
CSV_SALIDA = "datos_combinados.csv"
df[["fecha_hora", "T_interior", "T_exterior", "delta_T", "shortwave_radiation"]].to_csv(
    CSV_SALIDA, index=False, float_format="%.4f"
)
print(f"\n✔  DataFrame combinado exportado: {CSV_SALIDA}")
print(f"   Columnas: fecha_hora | T_interior | T_exterior | delta_T | shortwave_radiation")


# ════════════════════════════════════════════════════════════════════════════
# 5. SEPARAR DÍA / NOCHE
#    Umbral: radiación > 10 W/m²  →  día
#            radiación ≤ 10 W/m²  →  noche
# ════════════════════════════════════════════════════════════════════════════
UMBRAL_RAD = 0   # W/m²

df_dia   = df[df["shortwave_radiation"] >  UMBRAL_RAD].copy()
df_noche = df[df["shortwave_radiation"] <= UMBRAL_RAD].copy()

print(f"\nHoras de día  (rad > {UMBRAL_RAD} W/m²): {len(df_dia)}")
print(f"Horas de noche (rad ≤ {UMBRAL_RAD} W/m²): {len(df_noche)}")


# ════════════════════════════════════════════════════════════════════════════
# 6. CÁLCULO DE CORRELACIONES
# ════════════════════════════════════════════════════════════════════════════
def calcular_correlacion(df_sub, x_col, y_col, etiqueta):
    """Pearson r con test de significancia. Devuelve dict con resultados."""
    n = len(df_sub)
    if n < 3:
        print(f"\n[{etiqueta}] ⚠ Insuficientes datos ({n} filas). Omitido.")
        return None

    r, p = stats.pearsonr(df_sub[x_col].values, df_sub[y_col].values)
    r2   = r ** 2

    fuerza = (
        "fuerte"   if abs(r) >= 0.7 else
        "moderada" if abs(r) >= 0.4 else
        "débil"
    )
    signo = "positiva" if r > 0 else "negativa"
    sig   = "significativa (p < 0.05)" if p < 0.05 else "NO significativa (p ≥ 0.05)"

    print(f"\n{'═'*55}")
    print(f"  {etiqueta}")
    print(f"{'─'*55}")
    print(f"  n          = {n}")
    print(f"  Pearson r  = {r:+.4f}")
    print(f"  p-value    = {p:.4e}")
    print(f"  R²         = {r2:.4f}  ({r2*100:.1f}% de la varianza explicada)")
    print(f"  → Correlación {fuerza} {signo}, {sig}")
    print(f"{'═'*55}")

    return {"r": r, "p": p, "r2": r2, "n": n,
            "m": np.polyfit(df_sub[x_col], df_sub[y_col], 1)[0],
            "b": np.polyfit(df_sub[x_col], df_sub[y_col], 1)[1]}

res_dia   = calcular_correlacion(df_dia,   "shortwave_radiation",  "delta_T",    "DÍA   — Radiación (W/m²)  vs  ΔT = T_int − T_ext")
res_noche = calcular_correlacion(df_noche, "T_exterior", "T_interior", "NOCHE — T_exterior (°C)    vs  T_interior (°C)")


# ════════════════════════════════════════════════════════════════════════════
# 7. VISUALIZACIÓN
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

# ── Serie temporal ───────────────────────────────────────────────────────────
ax_ts = fig.add_subplot(gs[0, :])
ax_ts.plot(df["fecha_hora"], df["T_interior"], label="T interior (°C)",  color="tomato",    lw=1.2)
ax_ts.plot(df["fecha_hora"], df["T_exterior"], label="T exterior (°C)",  color="steelblue", lw=1.2)
ax_ts2 = ax_ts.twinx()
ax_ts2.fill_between(df["fecha_hora"], df["shortwave_radiation"], alpha=0.15, color="gold")
ax_ts2.plot(df["fecha_hora"], df["shortwave_radiation"], label="Radiación (W/m²)", color="goldenrod", lw=0.8)
ax_ts2.set_ylabel("Radiación solar (W/m²)", color="goldenrod")
ax_ts.set_ylabel("Temperatura (°C)")
ax_ts.set_title("Serie temporal completa")
lines1, labs1 = ax_ts.get_legend_handles_labels()
lines2, labs2 = ax_ts2.get_legend_handles_labels()
ax_ts.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8)

# ── Scatter DÍA ─────────────────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
if res_dia:
    ax_d.scatter(df_dia["shortwave_radiation"], df_dia["delta_T"],
                 alpha=0.45, color="orange", edgecolors="darkorange", s=25)
    x_line = np.linspace(df_dia["shortwave_radiation"].min(), df_dia["shortwave_radiation"].max(), 200)
    ax_d.plot(x_line, res_dia["m"] * x_line + res_dia["b"], "r--", lw=1.5,
              label=f"r = {res_dia['r']:+.3f}  |  R² = {res_dia['r2']:.3f}")
    ax_d.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax_d.set_xlabel("Radiación solar (W/m²)")
    ax_d.set_ylabel("ΔT = T_int − T_ext (°C)")
    ax_d.set_title("DÍA — Radiación vs ΔT")
    ax_d.legend(fontsize=8)

# ── Scatter NOCHE ────────────────────────────────────────────────────────────
ax_n = fig.add_subplot(gs[1, 1])
if res_noche:
    ax_n.scatter(df_noche["T_exterior"], df_noche["T_interior"],
                 alpha=0.45, color="steelblue", edgecolors="navy", s=25)
    x_line = np.linspace(df_noche["T_exterior"].min(), df_noche["T_exterior"].max(), 200)
    ax_n.plot(x_line, res_noche["m"] * x_line + res_noche["b"], "r--", lw=1.5,
              label=f"r = {res_noche['r']:+.3f}  |  R² = {res_noche['r2']:.3f}")
    ax_n.set_xlabel("T_exterior (°C)")
    ax_n.set_ylabel("T_interior (°C)")
    ax_n.set_title("NOCHE — T_exterior vs T_interior")
    ax_n.legend(fontsize=8)

fig.suptitle("Análisis de correlación — Invernadero", fontsize=13, fontweight="bold")
plt.savefig("correlaciones_invernadero.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✔  Gráfico guardado: correlaciones_invernadero.png")