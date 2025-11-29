"""
Script para procesar datos de entrenamiento con precios de competencia
"""
import pandas as pd
import numpy as np

print("📥 Cargando datos raw...")

# Cargar ventas
ventas = pd.read_csv('data/raw/entrenamiento/ventas.csv')
ventas['fecha'] = pd.to_datetime(ventas['fecha'])
print(f"✅ Ventas: {len(ventas)} registros")

# Cargar competencia
competencia = pd.read_csv('data/raw/entrenamiento/competencia.csv')
competencia['fecha'] = pd.to_datetime(competencia['fecha'])
print(f"✅ Competencia: {len(competencia)} registros")

# Merge
print("\n🔗 Haciendo merge...")
df = ventas.merge(competencia, on=['fecha', 'producto_id'], how='left')
print(f"✅ Merged: {len(df)} registros")

# Crear precio_competencia como promedio
print("\n💰 Calculando precio_competencia...")
df['precio_competencia'] = df[['Amazon', 'Decathlon', 'Deporvillage']].mean(axis=1)

# Calcular descuento_porcentaje
df['descuento_porcentaje'] = ((df['precio_venta'] - df['precio_base']) / df['precio_base']) * 100

# Calcular ratio_precio
df['ratio_precio'] = df['precio_venta'] / df['precio_competencia']

print(f"✅ precio_competencia - min: {df['precio_competencia'].min():.2f}, max: {df['precio_competencia'].max():.2f}")
print(f"✅ descuento_porcentaje - min: {df['descuento_porcentaje'].min():.2f}, max: {df['descuento_porcentaje'].max():.2f}")
print(f"✅ ratio_precio - min: {df['ratio_precio'].min():.3f}, max: {df['ratio_precio'].max():.3f}")

# Agregar features temporales básicas
print("\n📅 Agregando features temporales...")
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month
df['dia'] = df['fecha'].dt.day
df['dia_semana'] = df['fecha'].dt.dayofweek
df['fin_de_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
df['semana_año'] = df['fecha'].dt.isocalendar().week
df['trimestre'] = df['fecha'].dt.quarter
df['inicio_mes'] = (df['dia'] <= 10).astype(int)
df['fin_mes'] = (df['dia'] >= 20).astype(int)

# Festivos y eventos
df['es_festivo'] = 0
df['es_blackfriday'] = 0
df['es_cybermonday'] = 0
df['es_laborable'] = (~df['fin_de_semana'].astype(bool)).astype(int)

# Marcar Black Friday y Cyber Monday manualmente si los hay
# Aquí podrías agregar lógica específica si conoces las fechas

print("✅ Features temporales agregadas")

# Guardar
print("\n💾 Guardando df.csv...")
df.to_csv('data/processed/df.csv', index=False)
print(f"✅ Guardado: {len(df)} registros, {len(df.columns)} columnas")

print("\n📋 Columnas en df.csv:")
for col in df.columns:
    print(f"   - {col}")

print("\n✨ ¡Proceso completado!")
print("🚀 Ahora ejecuta: python regenerar_modelo.py")
