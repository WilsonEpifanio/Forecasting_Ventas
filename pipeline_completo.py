#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script completo de Forecasting de Ventas con correcciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import holidays
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PIPELINE DE FORECASTING DE VENTAS")
print("=" * 80)

# 1. CARGAR DATOS
print("\n[1/9] Cargando datos...")
ventas_df = pd.read_csv('data/raw/entrenamiento/ventas.csv')
competencia_df = pd.read_csv('data/raw/entrenamiento/competencia.csv')

# Convertir fechas
ventas_df['fecha'] = pd.to_datetime(ventas_df['fecha'])
competencia_df['fecha'] = pd.to_datetime(competencia_df['fecha'])

# Integrar dataframes
df = pd.merge(ventas_df, competencia_df, on=['fecha', 'producto_id'], suffixes=('_ventas', '_competencia'))
print(f"   Datos cargados: {len(df)} registros")

# 2. CREAR VARIABLES TEMPORALES
print("\n[2/9] Creando variables temporales...")
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['mes_nombre'] = df['fecha'].dt.month_name()
df['dia'] = df['fecha'].dt.day
df['dia_semana'] = df['fecha'].dt.dayofweek
df['nombre_dia_semana'] = df['fecha'].dt.day_name()
df['fin_de_semana'] = df['dia_semana'].isin([5, 6])

# Festivos
years = df['año'].unique()
es_holidays = holidays.country_holidays('ES', years=years)
df['es_festivo'] = df['fecha'].isin(es_holidays)
df['nombre_festivo'] = df['fecha'].map(lambda x: es_holidays.get(x, None))

# Black Friday
def es_blackfriday(fecha):
    if fecha.month == 11 and fecha.weekday() == 4:
        next_friday = fecha + pd.Timedelta(days=7)
        return next_friday.month != 11
    return False
df['es_blackfriday'] = df['fecha'].apply(es_blackfriday)

# Cyber Monday
def es_cybermonday(fecha):
    if fecha.month == 11 and fecha.weekday() == 0:
        bf_dates = [d for d in df['fecha'] if es_blackfriday(d) and d.year == fecha.year]
        if bf_dates:
            bf = max(bf_dates)
            return fecha > bf and (fecha - bf).days <= 3
    return False
df['es_cybermonday'] = df['fecha'].apply(es_cybermonday)

df['es_laborable'] = (~df['es_festivo']) & (~df['fin_de_semana'])
df['semana_año'] = df['fecha'].dt.isocalendar().week
df['trimestre'] = df['fecha'].dt.quarter
df['inicio_mes'] = df['dia'] <= 3
df['fin_mes'] = df['dia'] >= (df['fecha'] + pd.offsets.MonthEnd(0)).dt.day - 2
print("   Variables temporales creadas")

# 3. CREAR LAGS Y MEDIA MÓVIL
print("\n[3/9] Creando lags y media móvil...")
lags = range(1, 8)

def crear_lags_media(df_group):
    df_group = df_group.sort_values('fecha')
    for lag in lags:
        df_group[f'unidades_vendidas_lag{lag}'] = df_group['unidades_vendidas'].shift(lag)
    df_group['unidades_vendidas_mm7'] = df_group['unidades_vendidas'].rolling(window=7).mean()
    return df_group

lag_dfs = []
for (producto_id, año), grupo in df.groupby(['producto_id', 'año']):
    lag_dfs.append(crear_lags_media(grupo))
df_lags = pd.concat(lag_dfs)

cols_lag_mm = [f'unidades_vendidas_lag{i}' for i in lags] + ['unidades_vendidas_mm7']
df_lags = df_lags.dropna(subset=cols_lag_mm)
print(f"   Lags creados, registros: {len(df_lags)}")

# 4. CREAR VARIABLE DESCUENTO (CORREGIDA)
print("\n[4/9] Creando variable descuento_porcentaje (CORREGIDA)...")
print("   Fórmula: (precio_base - precio_venta) / precio_base * 100")
print("   → Positivo = Descuento, Negativo = Sobreprecio")
df_lags['descuento_porcentaje'] = (df_lags['precio_base'] - df_lags['precio_venta']) / df_lags['precio_base'] * 100

corr_desc = df_lags['descuento_porcentaje'].corr(df_lags['unidades_vendidas'])
print(f"   Correlación descuento ↔ unidades: {corr_desc:.4f}")
print(f"   Rango: [{df_lags['descuento_porcentaje'].min():.2f}, {df_lags['descuento_porcentaje'].max():.2f}]")

# 5. CREAR VARIABLES DE COMPETENCIA
print("\n[5/9] Creando variables de competencia...")
competidores = ['Amazon', 'Decathlon', 'Deporvillage']
df_lags['precio_competencia'] = df_lags[competidores].mean(axis=1)
df_lags['ratio_precio'] = df_lags['precio_venta'] / df_lags['precio_competencia']
df_lags = df_lags.drop(columns=competidores)
print("   Variables de competencia creadas")

# 6. ONE-HOT ENCODING
print("\n[6/9] Aplicando one-hot encoding...")
for col in ['nombre', 'categoria', 'subcategoria']:
    df_lags[f'{col}_h'] = df_lags[col]

variables_h = ['nombre_h', 'categoria_h', 'subcategoria_h']
df_lags = pd.get_dummies(df_lags, columns=variables_h, prefix=variables_h)
print(f"   Total de columnas después de one-hot: {len(df_lags.columns)}")

# 7. GUARDAR DATAFRAME
print("\n[7/9] Guardando dataframe transformado...")
ruta_guardado = 'data/processed/df.csv'
df_lags.to_csv(ruta_guardado, index=False)
print(f"   ✓ Guardado en {ruta_guardado}")
print(f"   Dimensiones: {df_lags.shape[0]} filas × {df_lags.shape[1]} columnas")

# 8. ENTRENAR MODELO
print("\n[8/9] Entrenando modelo...")
train_df = df_lags[df_lags['año'].isin([2021, 2022, 2023])]
test_df = df_lags[df_lags['año'] == 2024]

excluir = ['fecha', 'ingresos', 'unidades_vendidas']
X_cols = [col for col in train_df.columns if col not in excluir and train_df[col].dtype != 'O']

X_train = train_df[X_cols]
y_train = train_df['unidades_vendidas']
X_test = test_df[X_cols]
y_test = test_df['unidades_vendidas']

# Modelo de entrenamiento
model = HistGradientBoostingRegressor(
    learning_rate=0.05, max_iter=400, max_depth=7,
    l2_regularization=1.0, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"   Model Performance (Test):")
print(f"   - MAE:  {mae:.4f}")
print(f"   - RMSE: {rmse:.4f}")
print(f"   - R²:   {r2:.4f}")

# 9. MODELO FINAL Y IMPORTANCIA
print("\n[9/9] Entrenando modelo final y calculando importancia...")
X_full = df_lags[X_cols]
y_full = df_lags['unidades_vendidas']

modelo_final = HistGradientBoostingRegressor(
    learning_rate=0.05, max_iter=400, max_depth=7,
    l2_regularization=1.0, random_state=42
)
modelo_final.fit(X_full, y_full)

# Guardar modelo
ruta_modelo = 'models/modelo_final.joblib'
joblib.dump(modelo_final, ruta_modelo)
print(f"   ✓ Modelo guardado en {ruta_modelo}")

# Importancia de variables
print("\n   Calculando importancia de variables...")
result = permutation_importance(modelo_final, X_full, y_full, n_repeats=10, random_state=42, n_jobs=-1)
importancias = result.importances_mean
indices = np.argsort(importancias)[::-1]
variables_ordenadas = [X_cols[i] for i in indices]
importancias_ordenadas = importancias[indices]

# Mostrar top 10
print("\n   TOP 10 VARIABLES MÁS IMPORTANTES:")
print("   " + "=" * 70)
for i in range(min(10, len(variables_ordenadas))):
    var_name = variables_ordenadas[i]
    imp_value = importancias_ordenadas[i]
    
    # Marcar descuento_porcentaje
    marker = " ← DESCUENTO" if var_name == 'descuento_porcentaje' else ""
    print(f"   {i+1:2d}. {var_name:40s} {imp_value:8.6f}{marker}")

# Gráfico de importancia
plt.figure(figsize=(12, 14))
plt.barh(variables_ordenadas[:30], importancias_ordenadas[:30], color='skyblue')
plt.xlabel('Importancia media (Permutation Importance)')
plt.title('Top 30 Variables más Importantes - Modelo Final')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('models/importancia_variables.png', dpi=300, bbox_inches='tight')
print("\n   ✓ Gráfico guardado en models/importancia_variables.png")

# Resumen final
print("\n" + "=" * 80)
print("✓ PIPELINE COMPLETADO EXITOSAMENTE")
print("=" * 80)
print(f"\nResumen:")
print(f"  - Datos procesados: {df_lags.shape}")
print(f"  - Variables: {len(X_cols)}")
print(f"  - descuento_porcentaje rango: [{df_lags['descuento_porcentaje'].min():.2f}, {df_lags['descuento_porcentaje'].max():.2f}]")
print(f"  - descuento_porcentaje correlación: {corr_desc:.6f}")
print(f"  - Posición en ranking: {list(variables_ordenadas).index('descuento_porcentaje') + 1} de {len(variables_ordenadas)}")
print("\n")
