# 📊 Forecasting de Ventas - Simulador Interactivo

Aplicación de **predicción de ventas** construida con **Streamlit** y un modelo de **Machine Learning** (HistGradientBoostingRegressor) que permite simular escenarios de precios y demanda para productos de deportes y fitness.

## 🎯 Características

- 🔮 **Predicciones recursivas** día a día con actualización de lags
- 💰 **Análisis de sensibilidad** a cambios en precios de competencia
- 🛍️ **Detección automática** de eventos especiales (Black Friday, Cyber Monday)
- 📈 **Visualización interactiva** de resultados
- ⚙️ **Parámetros ajustables** (descuentos, escenarios de competencia)
- 📊 **18 features** para predicciones precisas

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/Forecasting_Ventas.git
cd Forecasting_Ventas

# Crear un entorno virtual (recomendado)
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Preparar los Datos

Los datos deben estar en `data/processed/`:
- `df.csv` - Datos de entrenamiento
- `inferencia_df_transformado.csv` - Datos para inferencia

Si necesitas regenerar el modelo con datos nuevos:

```bash
python regenerar_modelo.py
```

Esto:
- Carga los datos de entrenamiento
- Entrena un HistGradientBoostingRegressor
- Genera `models/modelo_final.joblib`
- Valida la compatibilidad con los datos de inferencia

### 3. Ejecutar la Aplicación

```bash
streamlit run app/app.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
Forecasting_Ventas/
├── app/
│   └── app.py                          # Aplicación Streamlit principal
├── data/
│   ├── raw/                            # Datos sin procesar
│   │   ├── entrenamiento/
│   │   │   ├── ventas.csv
│   │   │   └── competencia.csv
│   │   └── inferencia/
│   │       └── ventas_2025_inferencia.csv
│   └── processed/                      # Datos procesados
│       ├── df.csv                      # Datos para entrenamiento
│       └── inferencia_df_transformado.csv  # Datos para predicciones
├── models/
│   └── modelo_final.joblib             # Modelo entrenado (generado)
├── notebooks/
│   ├── entrenamiento.ipynb             # Pipeline de entrenamiento
│   └── forecasting.ipynb               # Análisis de predicciones
├── procesar_datos_entrenamiento.py     # Script de preparación de datos
├── regenerar_modelo.py                 # Script para entrenar el modelo
├── requirements.txt                    # Dependencias Python
└── README.md                           # Este archivo
```

## 🔧 Dependencias Principales

- **pandas** (>=2.0.0) - Manipulación de datos
- **scikit-learn** (>=1.3.0) - Modelo de ML
- **numpy** (<2.0.0) - Computación numérica
- **streamlit** (>=1.28.0) - Framework web
- **matplotlib** & **seaborn** - Visualización
- **joblib** (>=1.3.0) - Serialización de modelos

> ⚠️ **Nota importante**: numpy debe ser <2.0.0 para evitar conflictos de importación con scikit-learn

## 📊 Cómo Usar la Aplicación

1. **Seleccionar Producto**: Elige uno de los 25 productos disponibles
2. **Ajustar Descuento**: Modifica el precio con descuentos de -50% a +50%
3. **Escenario de Competencia**: Simula precios competitivos (actual, -5%, +5%)
4. **Ejecutar Simulación**: Presiona el botón "Simular"
5. **Analizar Resultados**:
   - Ver KPIs (unidades, ingresos, precio medio)
   - Gráfico de evolución diaria
   - Tabla detallada día por día
   - Comparativa de escenarios

## 🤖 Modelo de Machine Learning

### Tipo
**HistGradientBoostingRegressor** con 200 iteraciones

### Features (18 variables)
- Características del producto: `precio_base`, `es_estrella`
- Precios: `precio_competencia`, `descuento_porcentaje`, `ratio_precio`
- Temporales: `año`, `mes`, `dia`, `dia_semana`, `semana_año`, `trimestre`
- Calendarios: `fin_de_semana`, `es_festivo`, `es_blackfriday`, `es_cybermonday`, `es_laborable`
- Fechas especiales: `inicio_mes`, `fin_mes`

### Target
`unidades_vendidas` - Número de unidades predichas por día

## 🔄 Flujo de Predicción Recursivo

El modelo realiza predicciones día a día:
1. Para cada día, utiliza las 7 variables lag anteriores
2. Realiza la predicción para ese día
3. Actualiza los lags para el día siguiente
4. Itera hasta completar el mes

Esto permite capturar patrones secuenciales en las ventas.

## 📈 Próximos Pasos

- [ ] Integración con API de precios competitivos en tiempo real
- [ ] Análisis de elasticidad precio-demanda
- [ ] Dashboard de histórico de simulaciones
- [ ] Exportar resultados a Excel/PDF
- [ ] Predicción multi-producto

## 🌐 Despliegue en Streamlit Cloud

1. Sube este repositorio a GitHub
2. Accede a [streamlit.io](https://streamlit.io)
3. Haz clic en "Deploy an app"
4. Conecta tu repositorio de GitHub
5. Selecciona `app/app.py` como punto de entrada
6. ¡Listo! Tu app estará disponible en `https://tu-usuario-streamlit.streamlit.app`

## ⚙️ Troubleshooting

### Error: "numpy.core.multiarray failed to import"
```bash
pip uninstall numpy scikit-learn joblib
pip install "numpy<2.0.0" "scikit-learn>=1.3.0" "joblib>=1.3.0"
python regenerar_modelo.py
```

### El modelo tarda mucho en cargar
- Asegúrate de que `models/modelo_final.joblib` existe
- Verifica que Streamlit cache esté habilitado (@st.cache_resource)

### Las predicciones no cambian al ajustar parámetros
- Presiona el botón "🔄 Recargar" en la barra lateral
- Luego ajusta parámetros nuevamente

## 📝 Licencia

MIT - Libre para usar y modificar

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en GitHub.

---

**Última actualización**: Abril 2026
