# 📊 Risk Management 2025 - Portfolio Construction via Clustering

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

## 🎯 Objetivo

Construcción de carteras de inversión diversificadas utilizando técnicas de **clustering no supervisado** sobre más de 460 activos financieros (acciones y ETFs), con **backtesting** mensual comparado contra el benchmark **SPY (S&P 500)**.

## 📋 Metodología

1. ✅ **EDA**: Análisis exploratorio de datos de precios diarios (2019-2024)
2. ✅ **Feature Engineering**: Cálculo de 16 métricas de riesgo y retorno
   - Retornos (diarios y anualizados), Volatilidad
   - Sharpe Ratio, Sortino Ratio
   - Beta vs SPY, Correlación con benchmark, Alpha
   - VaR 95%, CVaR 95%, Maximum Drawdown
   - Skewness, Kurtosis
3. ✅ **Clustering**: Aplicación de técnicas avanzadas
   - **DBSCAN** para detección de outliers (29 activos de riesgo extremo)
   - **K-Means** (K=4, método seleccionado)
   - **Agglomerative Clustering** (Ward)
   - **Hierarchical Clustering** (dendrograma)
4. 🔄 **Portfolio Construction**: Selección de representantes por cluster *(próximo paso)*
5. 🔜 **Backtesting**: Simulación con rebalanceo mensual y costos de transacción

## 📊 Segmentación de Activos (Resultados Actuales)

El análisis de clustering identificó **5 segmentos** de activos:

| Segmento | Descripción | Activos | Criterio |
|----------|-------------|---------|----------|
| 🔴 **Outliers** | Riesgo Extremo | 29 (6.2%) | Detectados por DBSCAN |
| 🟢 **Alto Rendimiento** | Mejor Sharpe Ratio | ~96 | Mayor retorno ajustado por riesgo |
| 🔵 **Conservador** | Menor Volatilidad | ~50 | Menor riesgo total |
| 🟣 **Estable** | Menor Drawdown | ~199 | Caídas menos severas |
| 🟠 **Moderado** | Características intermedias | ~94 | Balance riesgo-retorno |

### Métricas de Clustering (K-Means, K=4)
- **Silhouette Score**: 0.2748
- **Davies-Bouldin Index**: 1.1119
- **Calinski-Harabasz Index**: 183.74

## 📁 Estructura del Proyecto

```
riskmanagement2025/
├── data/                         # Datos
│   ├── prices_train.csv          # Precios 2019-2023 (entrenamiento)
│   ├── prices_test.csv           # Precios 2024 (validación)
│   ├── features_matrix.csv       # Matriz de 16 features por activo
│   └── segmentacion_final/       # Resultados de clustering
│       ├── activos_segmentados_kmeans.csv
│       ├── resumen_segmentos.csv
│       ├── tickers_por_segmento.csv
│       └── metadata_segmentacion.txt
├── notebooks/                    # Jupyter notebooks (ejecutar en orden)
│   ├── 01_eda_data_loading.ipynb       ✅ Completado
│   ├── 02_feature_engineering.ipynb    ✅ Completado
│   ├── 03_clustering_analysis.ipynb    ✅ Completado
│   ├── 04_seleccion_portafolio.ipynb   🔄 Próximo
│   └── 05_backtesting.ipynb            🔜 Pendiente
├── src/                          # Módulos reutilizables
│   ├── data_loader.py
│   ├── features.py
│   ├── clustering.py
│   ├── portfolio.py
│   └── backtesting.py
├── reports/                      # Resultados y visualizaciones
│   ├── figures/                  # Gráficos generados
│   ├── clustering_results.csv
│   ├── prices_matrix.csv
│   ├── returns_matrix.csv
│   └── valid_tickers.csv
├── config/                       # Configuraciones
├── requirements.txt              # Dependencias
└── README.md
```

## 🚀 Cómo Ejecutar

### 1. Clonar repositorio
```bash
git clone https://github.com/kycido72/riskmanagement2025.git
cd riskmanagement2025
```

### 2. Crear entorno virtual (Anaconda)
```bash
conda create -n riskmanagement python=3.11
conda activate riskmanagement
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Datos
Colocar el archivo `trading_data.db` en la carpeta `data/`

> ⚠️ **Nota**: La base de datos no está incluida en el repositorio por razones de tamaño y privacidad.

### 5. Ejecutar notebooks
Ejecutar los notebooks en orden numérico (01 → 06)

```bash
jupyter notebook
```

## 📈 Resultados Preliminares

### Notebooks Completados

| Notebook | Descripción | Estado |
|----------|-------------|--------|
| `01_eda_data_loading` | Carga y exploración de datos | ✅ |
| `02_feature_engineering` | Cálculo de 16 métricas financieras | ✅ |
| `03_clustering_analysis` | Segmentación con K-Means, DBSCAN, Hierarchical | ✅ |
| `04_seleccion_portafolio` | Construcción de portafolios por perfil | 🔄 |
| `05_backtesting` | Simulación y comparación vs SPY | 🔜 |

### Datos Procesados
- **468 activos** analizados (acciones + ETFs)
- **5 años** de datos históricos (2019-2024)
- **16 features** de riesgo-retorno calculadas
- **5 segmentos** identificados mediante clustering

## ⚠️ Limitaciones

- Los datos históricos **no garantizan rendimiento futuro**
- Costos de transacción estimados (0.1%), no incluye slippage
- No considera restricciones de liquidez ni market impact
- Rebalanceo mensual asume ejecución al cierre del día
- No incluye dividendos en el cálculo de retornos

## 🛠️ Tecnologías

- **Python 3.11+**
- **Pandas / NumPy**: Manipulación de datos
- **Scikit-learn**: K-Means, Agglomerative, DBSCAN, PCA
- **HDBSCAN**: Detección de outliers
- **SciPy**: Hierarchical clustering, dendrogramas
- **Matplotlib / Seaborn**: Visualización
- **Jupyter Notebooks**: Análisis interactivo

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 👤 Autor

Juan Carlos Ruiz Arteaga, carlosarte11@gmail.com
---

*Proyecto desarrollado con fines educativos y de investigación. No constituye asesoría financiera.*
