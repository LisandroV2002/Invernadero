import pandas as pd
from sklearn.linear_model import LinearRegression

def calcular_modelo_termico(archivo_ext, archivo_int, archivo_meteo):
    print("Cargando y procesando datos...")
    
    # 1. Cargar datos de Open-Meteo
    # Usamos skiprows=3 porque Open-Meteo pone texto de metadatos en las primeras 3 líneas
    df_meteo = pd.read_csv(archivo_meteo, skiprows=3)
    df_meteo.columns = ['fecha_hora', 'shortwave_radiation']
    df_meteo['fecha_hora'] = pd.to_datetime(df_meteo['fecha_hora'])

    # 2. Cargar datos de sensores locales
    df_ext = pd.read_csv(archivo_ext)
    df_ext.columns = df_ext.columns.str.strip() # Limpiar espacios en los nombres de columnas
    df_ext['fecha_hora'] = pd.to_datetime(df_ext['fecha_hora'])

    df_inv = pd.read_csv(archivo_int)
    # Quitamos la zona horaria para poder cruzar los datos sin problemas
    df_inv['fecha_hora'] = pd.to_datetime(df_inv['fecha_hora']).dt.tz_localize(None)

    # 3. Calcular la Temperatura Interior Ponderada
    df_inv_t2m = df_inv[df_inv['nombre_sensor'] == 'TEMP_AMB_2M'][['fecha_hora', 'valor_medicion']].rename(columns={'valor_medicion': 'Temp_Interior_2m'})
    df_inv_t40cm = df_inv[df_inv['nombre_sensor'] == 'TEMP_AMB_40CM'][['fecha_hora', 'valor_medicion']].rename(columns={'valor_medicion': 'Temp_Interior_40cm'})

    df = pd.merge(df_ext, df_inv_t2m, on='fecha_hora', how='inner')
    df = pd.merge(df, df_inv_t40cm, on='fecha_hora', how='left')
    
    # Ponderación: 100% sensor de arriba (2m), 0% sensor de abajo (40cm)
    df['Temp_Int_Ponderada'] = df['Temp_Interior_2m'] * 1 + df['Temp_Interior_40cm'] * 0
    df.rename(columns={'temperatura_85': 'Temp_Exterior'}, inplace=True)

    # 4. Unir todo en una sola gran tabla
    df = pd.merge(df, df_meteo, on='fecha_hora', how='inner')

    # 5. Calcular la variable objetivo: Delta T
    df['Delta_T'] = df['Temp_Int_Ponderada'] - df['Temp_Exterior']

    # --- INICIO DEL ANÁLISIS MATEMÁTICO ---
    print("-" * 50)
    print("📊 RESULTADOS DEL MODELO TÉRMICO 📊")
    print("-" * 50)

    # A. Comportamiento Nocturno (Radiación 0 o muy baja)
    df_noche = df[df['shortwave_radiation'] <= 0]
    promedio_noche = df_noche['Delta_T'].mean()
    print(f"🌙 Comportamiento de Noche (Sin Sol):")
    print(f"   El invernadero se mantiene a un promedio de {promedio_noche:.2f} °C respecto al exterior.")

    # B. Comportamiento Diurno (Algoritmo de Regresión Lineal)
    df_dia = df[df['shortwave_radiation'] > 10].dropna(subset=['Delta_T', 'shortwave_radiation'])
    
    X = df_dia[['shortwave_radiation']] # Variable predictora
    y = df_dia['Delta_T']               # Variable a predecir

    modelo = LinearRegression()
    modelo.fit(X, y)

    coeficiente = modelo.coef_[0]
    intercepto = modelo.intercept_
    r2 = modelo.score(X, y)

    print(f"\n☀️ Comportamiento de Día (Motor Solar):")
    print(f"   Intercepto (b): {intercepto:.4f}")
    print(f"   Coeficiente (m): {coeficiente:.5f}")
    print(f"   Precisión del modelo (R²): {r2:.3f} (El sol explica el {r2*100:.0f}% del calor)")
    print("\n📈 REGLA DE ORO DE TU INVERNADERO:")
    print(f"   Por cada 100 W/m² de radiación, la temperatura sube {coeficiente * 100:.2f} °C.")

# Ejecutar la función con los nombres de tus archivos
# Nota: Cambiá los nombres si tus archivos se llaman distinto en el futuro
if __name__ == "__main__":
    calcular_modelo_termico(
        archivo_ext='temperatura_exterior.csv',
        archivo_int='datos_interior.csv',
        archivo_meteo='open-meteo-33.36S66.22W826m.csv'
    )