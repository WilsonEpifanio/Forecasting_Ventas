"""
Script para regenerar el modelo con versiones actuales de las librerías
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

print("🔄 Cargando datos de entrenamiento...")
try:
    df = pd.read_csv('data/processed/df.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    print(f"✅ Datos cargados: {len(df)} registros")
    print(f"📋 Columnas disponibles: {len(df.columns)}")
except Exception as e:
    print(f"❌ Error al cargar datos: {e}")
    exit(1)

# Verificar dataframe de inferencia para asegurar compatibilidad
print("\n🔍 Verificando compatibilidad con dataframe de inferencia...")
try:
    df_inf = pd.read_csv('data/processed/inferencia_df_transformado.csv')
    print(f"✅ Dataframe de inferencia: {len(df_inf)} registros, {len(df_inf.columns)} columnas")
except Exception as e:
    print(f"❌ Error al cargar inferencia: {e}")
    exit(1)

# Definir features y target
print("\n📊 Preparando features...")

# Columnas a excluir del modelo
columnas_excluir = [
    'fecha', 'producto_id', 'nombre', 'categoria', 'subcategoria',
    'unidades_vendidas', 'precio_venta', 'ingresos',
    'mes_nombre', 'nombre_dia_semana', 'nombre_festivo',
    'Amazon', 'Decathlon', 'Deporvillage'  # Excluir columnas que no están en inferencia
]

# IMPORTANTE: Verificar si las columnas clave existen en el dataframe
columnas_importantes = ['descuento_porcentaje', 'precio_competencia', 'ratio_precio']
columnas_faltantes_en_df = [col for col in columnas_importantes if col not in df.columns]

if columnas_faltantes_en_df:
    print(f"\n⚠️ ADVERTENCIA: Columnas importantes no encontradas en df.csv: {columnas_faltantes_en_df}")
    print("🔧 Creando estas columnas...")
    
    # Crear las columnas si no existen
    if 'descuento_porcentaje' not in df.columns and 'precio_venta' in df.columns and 'precio_base' in df.columns:
        df['descuento_porcentaje'] = ((df['precio_venta'] - df['precio_base']) / df['precio_base']) * 100
        print("  ✅ Creada: descuento_porcentaje")
    
    if 'precio_competencia' not in df.columns:
        # Si no existe, usar un precio de competencia simulado
        df['precio_competencia'] = df['precio_base'] * 0.95  # Ejemplo: 5% menos
        print("  ✅ Creada: precio_competencia (simulado)")
    
    if 'ratio_precio' not in df.columns and 'precio_venta' in df.columns and 'precio_competencia' in df.columns:
        df['ratio_precio'] = df['precio_venta'] / df['precio_competencia']
        print("  ✅ Creada: ratio_precio")

# Obtener todas las columnas disponibles
todas_columnas = df.columns.tolist()

# Features = todas las columnas menos las excluidas
features = [col for col in todas_columnas if col not in columnas_excluir]

# Verificar que las features están en el dataframe de inferencia
features_disponibles = []
for feat in features:
    if feat in df_inf.columns:
        features_disponibles.append(feat)
    else:
        print(f"⚠️ Feature '{feat}' no está en inferencia, se excluirá")

features = features_disponibles
print(f"✅ Total de features compatibles: {len(features)}")
print(f"📝 Features a usar:")
for f in features:
    print(f"   - {f}")

# Preparar X e y
X = df[features]
y = df['unidades_vendidas']

print(f"\n📈 Shape X: {X.shape}")
print(f"📈 Shape y: {y.shape}")

# Entrenar modelo
print("\n🤖 Entrenando modelo HistGradientBoostingRegressor...")
modelo = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.1,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    verbose=1
)

modelo.fit(X, y)

# Guardar modelo
print("\n💾 Guardando modelo...")
try:
    joblib.dump(modelo, 'models/modelo_final.joblib', compress=3)
    print("✅ Modelo guardado exitosamente en: models/modelo_final.joblib")
    
    # Verificar que se puede cargar
    print("\n🔍 Verificando modelo...")
    modelo_cargado = joblib.load('models/modelo_final.joblib')
    print(f"✅ Modelo verificado - Features: {len(modelo_cargado.feature_names_in_)}")
    
    # Hacer una predicción de prueba
    print("\n🧪 Predicción de prueba...")
    pred_test = modelo_cargado.predict(X.iloc[:1])
    print(f"✅ Predicción exitosa: {pred_test[0]:.2f}")
    
    print("\n✨ ¡Proceso completado con éxito!")
    print("🚀 Ahora puedes ejecutar la app de Streamlit")
    
except Exception as e:
    print(f"❌ Error al guardar modelo: {e}")
    exit(1)
