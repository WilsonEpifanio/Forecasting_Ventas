import pandas as pd
import numpy as np

# Cargar los datos
df_lags = pd.read_csv('data/processed/df.csv')

print(f'Total de columnas en el CSV: {len(df_lags.columns)}')
print(f'\nPrimeras 50 columnas:')
print(df_lags.columns.tolist()[:50])

print(f'\n--- Análisis de descuento_porcentaje ---')
if 'descuento_porcentaje' in df_lags.columns:
    print(f'Valores nulos: {df_lags["descuento_porcentaje"].isnull().sum()}')
    print(f'Valores infinitos: {np.isinf(df_lags["descuento_porcentaje"]).sum()}')
    print(f'Min: {df_lags["descuento_porcentaje"].min():.4f}')
    print(f'Max: {df_lags["descuento_porcentaje"].max():.4f}')
    print(f'Media: {df_lags["descuento_porcentaje"].mean():.4f}')
    print(f'Desv Estándar: {df_lags["descuento_porcentaje"].std():.4f}')

    # Correlación con unidades vendidas
    corr_descuento = df_lags[['descuento_porcentaje', 'unidades_vendidas']].corr().iloc[0, 1]
    print(f'\nCorrelación con unidades_vendidas: {corr_descuento:.6f}')
    
    print(f'\nMuestra de valores:')
    print(df_lags[['fecha', 'precio_base', 'precio_venta', 'descuento_porcentaje', 'unidades_vendidas']].head(10))
else:
    print('descuento_porcentaje NO ESTÁ en el CSV')
