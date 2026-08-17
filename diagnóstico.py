import pandas as pd
import numpy as np

# Cargar los datos
df_lags = pd.read_csv('data/processed/df.csv')

print('--- Análisis de descuento_porcentaje ---')
print(f'Columnas disponibles: {df_lags.columns.tolist()[:10]}...')
print(f'Total de columnas: {len(df_lags.columns)}')

if 'descuento_porcentaje' in df_lags.columns:
    print(f'\nValores nulos: {df_lags["descuento_porcentaje"].isnull().sum()}')
    print(f'Valores infinitos: {np.isinf(df_lags["descuento_porcentaje"]).sum()}')
    print(f'Min: {df_lags["descuento_porcentaje"].min():.2f}')
    print(f'Max: {df_lags["descuento_porcentaje"].max():.2f}')
    print(f'Media: {df_lags["descuento_porcentaje"].mean():.2f}')
    print(f'Correlación con unidades_vendidas: {df_lags["descuento_porcentaje"].corr(df_lags["unidades_vendidas"]):.4f}')
else:
    print('ADVERTENCIA: descuento_porcentaje no está en el CSV guardado')
