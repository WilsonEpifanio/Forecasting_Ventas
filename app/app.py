import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Obtener el directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data' / 'processed'

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Ventas Noviembre 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funciones auxiliares
@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado"""
    try:
        modelo_path = MODELS_DIR / 'modelo_final.joblib'
        modelo = joblib.load(str(modelo_path))
        st.sidebar.success(f"✅ Modelo cargado: {len(modelo.feature_names_in_)} features")
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.error(f"📁 Ruta buscada: {MODELS_DIR / 'modelo_final.joblib'}")
        st.error("💡 Intenta regenerar el modelo con la versión actual de las librerías")
        return None

@st.cache_data
def cargar_datos():
    """Carga el dataframe de inferencia transformado"""
    try:
        datos_path = DATA_DIR / 'inferencia_df_transformado.csv'
        df = pd.read_csv(str(datos_path))
        df['fecha'] = pd.to_datetime(df['fecha'])
        st.sidebar.success(f"✅ Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        st.error(f"📁 Ruta buscada: {DATA_DIR / 'inferencia_df_transformado.csv'}")
        return None

def aplicar_ajustes(df_producto, descuento_ajuste, escenario_competencia):
    """Aplica los ajustes de descuento y competencia al dataframe"""
    df_ajustado = df_producto.copy()
    
    # Ajustar precio_venta según el descuento
    factor_descuento = 1 + (descuento_ajuste / 100)
    df_ajustado['precio_venta'] = df_ajustado['precio_base'] * factor_descuento
    
    # Ajustar precio_competencia según el escenario
    factor_competencia = 1.0
    if escenario_competencia == "Competencia -5%":
        factor_competencia = 0.95
    elif escenario_competencia == "Competencia +5%":
        factor_competencia = 1.05
    
    # Aplicar factor a precio_competencia (que ya es el promedio en el dataframe)
    df_ajustado['precio_competencia'] = df_ajustado['precio_competencia'] * factor_competencia
    
    # Recalcular descuento_porcentaje y ratio_precio
    df_ajustado['descuento_porcentaje'] = ((df_ajustado['precio_venta'] - df_ajustado['precio_base']) / df_ajustado['precio_base']) * 100
    df_ajustado['ratio_precio'] = df_ajustado['precio_venta'] / df_ajustado['precio_competencia']
    
    return df_ajustado

def predecir_recursivo(modelo, df_ajustado):
    """Realiza predicciones recursivas día por día actualizando lags"""
    df_prediccion = df_ajustado.copy()
    df_prediccion = df_prediccion.sort_values('fecha').reset_index(drop=True)
    
    # Obtener las columnas que el modelo espera
    feature_names = modelo.feature_names_in_
    
    # Verificar que todas las features están en el dataframe
    columnas_faltantes = [col for col in feature_names if col not in df_prediccion.columns]
    if columnas_faltantes:
        st.error(f"❌ Columnas faltantes en el dataframe: {columnas_faltantes}")
        st.error(f"🔍 Features del modelo: {list(feature_names)}")
        st.error(f"🔍 Columnas disponibles: {list(df_prediccion.columns)}")
        st.stop()
    
    # Array para almacenar predicciones
    predicciones = []
    
    # Nombres de las columnas de lag
    lag_cols = [f'unidades_vendidas_lag_{i}' for i in range(1, 8)]
    
    for idx in range(len(df_prediccion)):
        # Preparar features para este día
        X_dia = df_prediccion.iloc[[idx]][feature_names]
        
        # Hacer predicción
        pred = modelo.predict(X_dia)[0]
        predicciones.append(pred)
        
        # Actualizar lags para el siguiente día (si no es el último)
        if idx < len(df_prediccion) - 1:
            # Desplazar lags: lag_7 <- lag_6, lag_6 <- lag_5, ..., lag_2 <- lag_1
            for i in range(7, 1, -1):
                if f'unidades_vendidas_lag_{i}' in df_prediccion.columns:
                    df_prediccion.loc[idx + 1, f'unidades_vendidas_lag_{i}'] = df_prediccion.loc[idx, f'unidades_vendidas_lag_{i-1}']
            
            # Actualizar lag_1 con la predicción actual
            if 'unidades_vendidas_lag_1' in df_prediccion.columns:
                df_prediccion.loc[idx + 1, 'unidades_vendidas_lag_1'] = pred
            
            # Actualizar media móvil de 7 días
            if 'unidades_vendidas_ma7' in df_prediccion.columns:
                # Obtener las últimas 7 predicciones (o las que haya disponibles)
                ultimas_predicciones = predicciones[-min(7, len(predicciones)):]
                df_prediccion.loc[idx + 1, 'unidades_vendidas_ma7'] = np.mean(ultimas_predicciones)
    
    df_prediccion['unidades_predichas'] = predicciones
    df_prediccion['ingresos_proyectados'] = df_prediccion['unidades_predichas'] * df_prediccion['precio_venta']
    
    return df_prediccion

def crear_grafico_prediccion(df_resultado):
    """Crea el gráfico de predicción diaria"""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Configurar seaborn con estilo minimalista
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.1)
    
    # Extraer día del mes
    df_resultado['dia_mes'] = df_resultado['fecha'].dt.day
    
    # Línea principal de predicción - color profesional
    sns.lineplot(
        data=df_resultado,
        x='dia_mes',
        y='unidades_predichas',
        ax=ax,
        color='#2563EB',
        linewidth=2.5,
        marker='o',
        markersize=5
    )
    
    # Marcar Black Friday (día 28 de noviembre)
    black_friday_dia = 28
    if black_friday_dia in df_resultado['dia_mes'].values:
        bf_row = df_resultado[df_resultado['dia_mes'] == black_friday_dia].iloc[0]
        
        # Línea vertical sutil
        ax.axvline(x=black_friday_dia, color='#DC2626', linestyle='--', linewidth=1.5, alpha=0.6)
        
        # Punto destacado
        ax.scatter([black_friday_dia], [bf_row['unidades_predichas']], 
                  color='#DC2626', s=150, zorder=5, edgecolors='white', linewidth=2)
        
        # Anotación minimalista
        ax.annotate(
            f'Black Friday\n{bf_row["unidades_predichas"]:.0f} unidades',
            xy=(black_friday_dia, bf_row['unidades_predichas']),
            xytext=(black_friday_dia - 3, bf_row['unidades_predichas'] * 1.12),
            fontsize=9,
            fontweight='600',
            color='#374151',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#E5E7EB', linewidth=1),
            arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.5)
        )
    
    # Etiquetas limpias
    ax.set_xlabel('Día de Noviembre 2025', fontsize=11, fontweight='600', color='#374151')
    ax.set_ylabel('Unidades Predichas', fontsize=11, fontweight='600', color='#374151')
    ax.set_title('Predicción de Ventas Diarias', fontsize=13, fontweight='700', color='#111827', pad=15)
    
    # Grid sutil
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#E5E7EB')
    ax.set_axisbelow(True)
    
    # Configurar eje X
    ax.set_xticks(range(1, 31, 3))
    
    # Quitar bordes superiores y derechos (estilo minimalista)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E7EB')
    ax.spines['bottom'].set_color('#E5E7EB')
    
    # Color de ticks
    ax.tick_params(colors='#6B7280', labelsize=9)
    
    plt.tight_layout()
    return fig

# Cargar datos y modelo
modelo = cargar_modelo()
df_inferencia = cargar_datos()

if modelo is None or df_inferencia is None:
    st.stop()

# Obtener lista de productos únicos
productos = sorted(df_inferencia['nombre'].unique())

# ==================== SIDEBAR ====================
st.sidebar.markdown("# 🎛️ Controles")

# Botón para limpiar caché
if st.sidebar.button("🔄 Recargar", help="Limpia el caché y recarga el modelo"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("")

# Título de parámetros
st.sidebar.markdown("# 🎛️ Parámetros")
st.sidebar.markdown("")

# Selector de producto
producto_seleccionado = st.sidebar.selectbox(
    "Producto",
    options=productos,
    index=0
)

# Slider de descuento
descuento_ajuste = st.sidebar.slider(
    "Ajuste de Descuento",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    format="%d%%",
    help="Modifica el precio base del producto"
)

# Selector de escenario de competencia
st.sidebar.markdown("### Escenario de Competencia")
escenario_competencia = st.sidebar.radio(
    "Precios de la competencia",
    options=["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    index=0,
    label_visibility="collapsed"
)

st.sidebar.markdown("")

# Botón de simulación
simular = st.sidebar.button("▶ Simular", type="primary")

st.sidebar.markdown("---")
st.sidebar.info("💡 Ajusta los parámetros y ejecuta la simulación")
st.sidebar.info("💡 Una vez realizada la simulación y quiera hacer otra, pulse el botón Recargar, ajuste los parámetros que desea y pulse el botón Simular")

# ==================== ZONA PRINCIPAL ====================

# Header con espaciado
st.title("📊 Simulación de Ventas")
st.markdown(f"<h3 style='color: #FFFFFF; font-weight: 600; margin-top: 0rem;'>{producto_seleccionado} • Noviembre 2025</h3>", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

if simular:
    with st.spinner('Generando predicciones...'):
        # Filtrar datos del producto seleccionado
        df_producto = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
        
        if df_producto.empty:
            st.error("No hay datos disponibles para el producto seleccionado")
            st.stop()
        
        # Aplicar ajustes
        df_ajustado = aplicar_ajustes(df_producto, descuento_ajuste, escenario_competencia)
        
        # Realizar predicciones recursivas
        df_resultado = predecir_recursivo(modelo, df_ajustado)
        
        # ==================== KPIs ====================
        st.markdown("### Resumen")
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        
        unidades_totales = df_resultado['unidades_predichas'].sum()
        ingresos_totales = df_resultado['ingresos_proyectados'].sum()
        precio_promedio = df_resultado['precio_venta'].mean()
        descuento_promedio = df_resultado['descuento_porcentaje'].mean()
        
        col1, col2, col3, col4 = st.columns(4, gap="medium")
        
        with col1:
            st.metric(
                label="Unidades",
                value=f"{unidades_totales:,.0f}"
            )
        
        with col2:
            st.metric(
                label="Ingresos",
                value=f"€{ingresos_totales:,.0f}"
            )
        
        with col3:
            st.metric(
                label="Precio Medio",
                value=f"€{precio_promedio:.2f}"
            )
        
        with col4:
            st.metric(
                label="Descuento",
                value=f"{descuento_promedio:.1f}%"
            )
        
        st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)
        
        # ==================== GRÁFICO ====================
        st.markdown("### Evolución Diaria")
        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
        fig = crear_grafico_prediccion(df_resultado)
        st.pyplot(fig)
        
        st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)
        
        # ==================== TABLA DETALLADA ====================
        st.markdown("### Detalle Diario")
        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
        
        # Preparar tabla
        df_tabla = df_resultado[['fecha', 'precio_venta', 'precio_competencia', 
                                 'descuento_porcentaje', 'unidades_predichas', 
                                 'ingresos_proyectados']].copy()
        
        df_tabla['dia_semana'] = df_tabla['fecha'].dt.day_name()
        df_tabla['fecha_display'] = df_tabla['fecha'].dt.strftime('%d/%m/%Y')
        
        # Identificar Black Friday
        df_tabla['es_black_friday'] = df_tabla['fecha'].dt.day == 28
        
        # Reordenar columnas
        df_tabla = df_tabla[['fecha_display', 'dia_semana', 'precio_venta', 
                             'precio_competencia', 'descuento_porcentaje', 
                             'unidades_predichas', 'ingresos_proyectados', 'es_black_friday']]
        
        df_tabla.columns = ['Fecha', 'Día', 'Precio (€)', 
                           'Competencia (€)', 'Desc. (%)', 
                           'Unidades', 'Ingresos (€)', 'BF']
        
        # Formatear números
        df_tabla['Precio (€)'] = df_tabla['Precio (€)'].apply(lambda x: f"{x:.2f}")
        df_tabla['Competencia (€)'] = df_tabla['Competencia (€)'].apply(lambda x: f"{x:.2f}")
        df_tabla['Desc. (%)'] = df_tabla['Desc. (%)'].apply(lambda x: f"{x:.1f}")
        df_tabla['Unidades'] = df_tabla['Unidades'].apply(lambda x: f"{x:.0f}")
        df_tabla['Ingresos (€)'] = df_tabla['Ingresos (€)'].apply(lambda x: f"{x:.2f}")
        df_tabla['BF'] = df_tabla['BF'].apply(lambda x: '🛍️' if x else '')
        
        st.dataframe(
            df_tabla,
            height=350,
            hide_index=True
        )
        
        st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)
        
        # ==================== COMPARATIVA DE ESCENARIOS ====================
        st.markdown("## Análisis de Escenarios")
        st.markdown("<p style='font-size: 1.1rem; color: #FFFFFF; margin-top: -0.5rem;'>Sensibilidad a cambios en precios de competencia</p>", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        
        escenarios = {
            "Sin Cambios": 0,
            "Competencia -5%": -5,
            "Competencia +5%": +5
        }
        
        resultados_escenarios = {}
        
        # Para debugging - mostrar valores originales
        df_original_producto = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
        precio_comp_original = df_original_producto['precio_competencia'].iloc[0]
        
        with st.spinner('Calculando escenarios...'):
            for nombre_esc, ajuste_comp in escenarios.items():
                # IMPORTANTE: Crear una copia fresca del dataframe ORIGINAL del producto
                # para que cada escenario parta del mismo estado inicial
                df_esc = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
                
                # Aplicar descuento del usuario
                factor_descuento = 1 + (descuento_ajuste / 100)
                df_esc['precio_venta'] = df_esc['precio_base'] * factor_descuento
                
                # Aplicar ajuste de competencia del escenario
                factor_comp = 1 + (ajuste_comp / 100)
                df_esc['precio_competencia'] = df_original_producto['precio_competencia'].values * factor_comp
                
                # Recalcular variables derivadas
                df_esc['descuento_porcentaje'] = ((df_esc['precio_venta'] - df_esc['precio_base']) / df_esc['precio_base']) * 100
                df_esc['ratio_precio'] = df_esc['precio_venta'] / df_esc['precio_competencia']
                
                # Predecir con lags originales para cada escenario
                df_pred_esc = predecir_recursivo(modelo, df_esc)
                
                # Calcular resultados
                unidades_totales = df_pred_esc['unidades_predichas'].sum()
                ingresos_totales = df_pred_esc['ingresos_proyectados'].sum()
                
                resultados_escenarios[nombre_esc] = {
                    'unidades': unidades_totales,
                    'ingresos': ingresos_totales,
                    'precio_comp_ejemplo': df_esc['precio_competencia'].iloc[0],  # Para debug
                    'ratio_precio_ejemplo': df_esc['ratio_precio'].iloc[0]  # Para debug
                }
        
        # Mostrar resultados en columnas
        cols = st.columns(3, gap="large")
        
        for idx, (nombre_esc, resultados) in enumerate(resultados_escenarios.items()):
            with cols[idx]:
                st.markdown(f"<h3 style='font-weight: 600; color: #FFFFFF; margin-bottom: 1rem; font-size: 1.2rem;'>{nombre_esc}</h3>", unsafe_allow_html=True)
                st.metric("Unidades", f"{resultados['unidades']:,.0f}")
                st.metric("Ingresos", f"€{resultados['ingresos']:,.0f}")
                
                # Mostrar debug info colapsado
                with st.expander("Detalles técnicos"):
                    st.caption(f"Precio Comp. Base: €{precio_comp_original:.2f}")
                    st.caption(f"Precio Comp. Ajustado: €{resultados['precio_comp_ejemplo']:.2f}")
                    st.caption(f"Ratio: {resultados['ratio_precio_ejemplo']:.3f}")

else:
    st.info("Configura los parámetros en el panel lateral y ejecuta la simulación")
    
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Información")
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.metric("Productos", len(productos))
    
    with col2:
        st.metric("Días", "30")
    
    with col3:
        st.metric("Mes", "Nov 2025")