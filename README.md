# 🏦 Sistema de Gestión de Portafolios con Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Completado-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistema de selección y construcción de portafolios de inversión basado en clustering K-Means, diseñado para clasificar activos financieros según características cuantitativas y adaptar las recomendaciones al perfil de riesgo del inversionista.

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Metodología](#-metodología)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Pipeline de Producción](#-pipeline-de-producción)
- [Resultados](#-resultados)
- [Notebooks](#-notebooks)
- [Roadmap](#-roadmap)
- [Autor](#-autor)
- [Disclaimer](#-disclaimer)

---

## 📖 Descripción

Este proyecto implementa un sistema cuantitativo de gestión de portafolios que utiliza técnicas de Machine Learning (K-Means Clustering) para:

1. **Segmentar activos financieros** según características como retorno, volatilidad, momentum y beta
2. **Clasificar inversionistas** en 5 perfiles de riesgo
3. **Construir portafolios optimizados** según el perfil del inversionista
4. **Realizar backtesting** para evaluar el desempeño histórico

### Universo de Activos
- **Fuente**: S&P 500 + ETFs principales
- **Período de análisis**: 2019-2024 (datos de entrenamiento hasta 2023)
- **Período de backtesting**: 2024 (out-of-sample)
- **Activos válidos**: 472 tickers con datos completos

---

## 🔬 Metodología

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DE ANÁLISIS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐│
│  │    DATOS     │───▶│  FEATURES    │───▶│  CLUSTERING  │───▶│  PORTAFOLIO││
│  │              │    │              │    │              │    │            ││
│  │ • S&P 500    │    │ • Retorno    │    │ • K-Means    │    │ • Scoring  ││
│  │ • ETFs      │    │ • Volatilidad│    │ • K=5        │    │ • Top N    ││
│  │ • 2019-2024 │    │ • Momentum   │    │ • Silhouette │    │ • Equal Wt ││
│  │              │    │ • Beta       │    │              │    │            ││
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 📊 Feature Engineering

| Feature | Descripción | Cálculo |
|---------|-------------|---------|
| **Retorno Anual** | Rendimiento compuesto anualizado | $(1 + r_{total})^{252/n} - 1$ |
| **Volatilidad** | Desviación estándar anualizada | $\sigma_{daily} \times \sqrt{252}$ |
| **Sharpe Ratio** | Retorno ajustado por riesgo | $(r - r_f) / \sigma$ |
| **Momentum 6M** | Rendimiento últimos 6 meses | $P_t / P_{t-126} - 1$ |
| **Beta** | Sensibilidad al mercado (SPY) | $Cov(r_i, r_m) / Var(r_m)$ |

### 🎯 Fórmula de Scoring

El sistema utiliza una fórmula de score compuesto para rankear los activos dentro de cada segmento:

$$Score = 0.35 \times Return_{norm} + 0.30 \times Momentum_{6m,norm} + 0.15 \times Sharpe_{norm} + 0.20 \times Beta_{adj}$$

**Explicación de los componentes:**
- **Return (35%)**: Factor dominante que premia activos con mayores retornos históricos
- **Momentum 6M (30%)**: Captura tendencias recientes, favoreciendo activos con impulso positivo
- **Sharpe Ratio (15%)**: Ajuste por riesgo que penaliza volatilidad excesiva
- **Beta Ajustado (20%)**: Se ajusta según el perfil del inversionista (mayor peso para agresivos, inverso para conservadores)

> Los valores se normalizan usando Min-Max scaling dentro de cada segmento para mantener comparabilidad.

### 🔢 Segmentación por K-Means

El algoritmo K-Means agrupa los activos en **5 clusters** basados en sus características:

| Cluster | Nombre | Características | # Activos |
|---------|--------|-----------------|-----------|
| 0 | **Outliers** | Rendimientos extremos, alta volatilidad, posibles anomalías | 24 |
| 1 | **Alto Rendimiento** | Retornos superiores, momentum fuerte, beta elevado | 94 |
| 2 | **Conservador** | Baja volatilidad, rendimientos modestos, beta bajo | 96 |
| 3 | **Estable** | Rendimientos consistentes, volatilidad moderada | 115 |
| 4 | **Moderado** | Balance riesgo-retorno equilibrado | 143 |

### 👤 Perfiles de Inversionista

El sistema mapea automáticamente clusters a perfiles de riesgo:

| Perfil | Clusters Asignados | Activos Disponibles | Descripción |
|--------|-------------------|---------------------|-------------|
| 🟢 **Conservador** | 2, 3 | 211 | Prioriza preservación de capital |
| 🔵 **Moderado** | 3, 4 | 258 | Balance entre crecimiento y estabilidad |
| ⚪ **Normal** | 2, 3, 4 | 354 | Perfil balanceado, diversificación amplia |
| 🟠 **Agresivo** | 1, 4 | 237 | Busca crecimiento con mayor volatilidad |
| 🔴 **Especulativo** | 0, 1 | 118 | Máximo riesgo por máximo retorno potencial |

---

## 📁 Estructura del Proyecto

```
riskmanagement2025/
│
├── 📊 data/
│   ├── prices_train.csv              # Precios históricos 2019-2023
│   ├── prices_test.csv               # Precios 2024 (backtesting)
│   ├── features_matrix.csv           # Features calculados por activo
│   └── segmentacion_final/
│       ├── activos_segmentados_kmeans.csv   # Activos con cluster asignado
│       ├── tickers_por_segmento.csv         # Lista de tickers por cluster
│       ├── resumen_segmentos.csv            # Estadísticas por cluster
│       └── metadata_segmentacion.txt        # Parámetros del modelo
│
├── 📓 notebooks/
│   ├── 01_eda_data_loading.ipynb     # Carga y exploración de datos
│   ├── 02_feature_engineering.ipynb  # Construcción de features
│   ├── 03_clustering_analysis.ipynb  # Análisis de clusters K-Means
│   ├── 04_portfolio_selection.ipynb  # Selección y backtesting
│   └── 05_reporte_final.ipynb        # Dashboard de resultados
│
├── 📈 reports/
│   ├── valid_tickers.csv             # Tickers válidos para análisis
│   ├── prices_matrix.csv             # Matriz de precios procesada
│   ├── returns_matrix.csv            # Matriz de retornos diarios
│   ├── clustering_results.csv        # Resultados del clustering
│   ├── portafolio_*.csv              # Portafolios por perfil
│   ├── backtest_*.csv                # Resultados de backtest
│   └── figures/                      # Gráficos generados
│
├── 🔧 src/
│   ├── __init__.py
│   ├── data_loader.py                # Funciones de carga de datos
│   ├── features.py                   # Cálculo de features
│   ├── clustering.py                 # Algoritmos de clustering
│   ├── portfolio.py                  # Construcción de portafolios
│   ├── backtesting.py                # Motor de backtesting
│   └── utils.py                      # Utilidades comunes
│
├── 🔄 pipeline/                       # Pipeline de producción
│   ├── __init__.py
│   ├── run_pipeline.py               # Orquestador CLI principal
│   ├── 01_data_ingestion.py          # Paso 1: Carga de datos
│   ├── 02_feature_engineering.py     # Paso 2: Cálculo de features
│   ├── 03_clustering.py              # Paso 3: Segmentación
│   ├── 04_portfolio_selection.py     # Paso 4: Portafolios + backtest
│   └── 05_generate_reports.py        # Paso 5: Outputs para web
│
├── 📤 outputs/
│   └── api/                          # Archivos para aplicación web
│       ├── portfolios.csv
│       ├── segments.csv
│       ├── backtest_summary.csv
│       ├── equity_curves.csv
│       └── metadata.json
│
├── ⚙️ config/
│   ├── settings.yaml                 # Configuración del pipeline
│   └── profiles.yaml                 # Definición de perfiles
│
├── requirements.txt                  # Dependencias del proyecto
├── LICENSE                           # Licencia MIT
└── README.md                         # Este archivo
```

---

## 🚀 Instalación

### Prerrequisitos
- Python 3.11+
- Conda (recomendado) o pip

### Pasos de Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/kycido72/riskmanagement2025.git
cd riskmanagement2025

# 2. Crear ambiente virtual con Conda
conda create -n riskmanagementportfolio python=3.11
conda activate riskmanagementportfolio

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar notebooks en orden
jupyter notebook
```

### Dependencias Principales
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
yfinance>=0.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## � Pipeline de Producción

El proyecto incluye un pipeline automatizado para entornos de producción, diseñado para alimentar una aplicación web.

### Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DE PRODUCCIÓN                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐│
│  │  01_DATA     │───▶│  02_FEATURES │───▶│  03_CLUSTER  │───▶│04_PORTFOLIO││
│  │  INGESTION   │    │  ENGINEERING │    │              │    │ SELECTION  ││
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘│
│         │                                                           │       │
│         │                                                           ▼       │
│         │                                                    ┌────────────┐│
│         │                                                    │05_REPORTS  ││
│         │                                                    │ (API CSV)  ││
│         │                                                    └────────────┘│
│         │                                                           │       │
│         ▼                                                           ▼       │
│   data/*.csv                                              outputs/api/*.csv │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Uso del Pipeline

```bash
# Ejecutar pipeline completo
python -m pipeline.run_pipeline --all

# Ejecutar etapas específicas
python -m pipeline.run_pipeline --stages 1,2,3    # Solo data + features + clustering
python -m pipeline.run_pipeline --stages 4,5      # Solo portafolios + reportes

# Reentrenamiento (etapas 2-5, asume datos existentes)
python -m pipeline.run_pipeline --retrain

# Ver estado del pipeline
python -m pipeline.run_pipeline --status
```

### Archivos de Salida para Web App

| Archivo | Contenido |
|---------|-----------|
| `outputs/api/portfolios.csv` | Composición de todos los portafolios |
| `outputs/api/segments.csv` | Información de segmentos de activos |
| `outputs/api/backtest_summary.csv` | Métricas consolidadas de backtest |
| `outputs/api/equity_curves.csv` | Series temporales de equity |
| `outputs/api/metadata.json` | Metadatos del pipeline |

### Configuración

Los parámetros del pipeline se configuran en `config/settings.yaml`:

```yaml
# Parámetros financieros
financial_params:
  risk_free_rate: 0.05
  trading_days: 252

# Clustering
clustering:
  n_clusters: 4
  outlier_detection: true

# Reentrenamiento
retraining:
  frequency_months: 6
```

---

## �📊 Resultados

### Backtesting 2024 (Out-of-Sample)

Resultados del backtesting con capital inicial de **$10,000 USD**:

| Perfil | Retorno Total | Sharpe Ratio | Max Drawdown | Capital Final |
|--------|---------------|--------------|--------------|---------------|
| 🟢 Conservador | 19.85% | 0.92 | -8.45% | $11,985 |
| 🔵 Moderado | 25.32% | 1.08 | -11.23% | $12,532 |
| ⚪ Normal | 27.41% | 1.15 | -12.67% | $12,741 |
| 🟠 Agresivo | 34.56% | 1.21 | -15.89% | $13,456 |
| 🔴 Especulativo | 42.18% | 0.98 | -22.34% | $14,218 |
| 📊 Benchmark (SPY) | 24.89% | 1.12 | -8.50% | $12,489 |

> **Nota**: Los perfiles Agresivo y Especulativo superaron al benchmark (SPY) en términos de retorno, mientras que el perfil Conservador ofreció mejor protección ante drawdowns.

### Métricas del Modelo

- **Silhouette Score**: 0.42 (clustering de calidad aceptable)
- **Número de Clusters**: 5 (óptimo según método del codo)
- **Activos Totales**: 472 tickers válidos
- **Período de Training**: 2019-2023
- **Período de Testing**: 2024

---

## 📓 Notebooks

| # | Notebook | Descripción | Estado |
|---|----------|-------------|--------|
| 01 | [EDA & Data Loading](notebooks/01_eda_data_loading.ipynb) | Carga de datos y análisis exploratorio | ✅ Completado |
| 02 | [Feature Engineering](notebooks/02_feature_engineering.ipynb) | Cálculo de features cuantitativos | ✅ Completado |
| 03 | [Clustering Analysis](notebooks/03_clustering_analysis.ipynb) | Segmentación K-Means de activos | ✅ Completado |
| 04 | [Portfolio Selection](notebooks/04_portfolio_selection.ipynb) | Selección de activos y backtesting | ✅ Completado |
| 05 | [Reporte Final](notebooks/05_reporte_final.ipynb) | Dashboard interactivo de resultados | ✅ Completado |

---

## 🗺️ Roadmap

### Fase 1: MVP ✅
- [x] Pipeline de datos automatizado
- [x] Feature engineering robusto
- [x] Clustering K-Means con validación
- [x] Sistema de scoring por perfil
- [x] Backtesting out-of-sample
- [x] Dashboard de resultados

### Fase 2: Producción ✅
- [x] Pipeline modular reproducible (`pipeline/`)
- [x] Configuración YAML centralizada (`config/`)
- [x] CLI para ejecución de etapas
- [x] Outputs CSV para aplicación web (`outputs/api/`)
- [x] Soporte para reentrenamiento cada 6 meses
- [ ] API REST para consulta de portafolios
- [ ] Sistema de rebalanceo automático

### Fase 3: Avanzado 📋
- [ ] Optimización por Markowitz
- [ ] Modelos de ML adicionales (Random Forest, XGBoost)
- [ ] Análisis de sentimiento
- [ ] Factores ESG
- [ ] Cobertura con opciones

---

## 👨‍💻 Autor

**Juan Carlos Ruiz Arteaga**

- GitHub: [@kycido72](https://github.com/kycido72)
- Proyecto desarrollado para el curso de Gestión de Riesgo 2025

---

## ⚠️ Disclaimer

> **ADVERTENCIA LEGAL**: Este proyecto es únicamente con fines educativos y de investigación. 
> 
> - No constituye asesoría de inversión ni recomendación de compra o venta de valores
> - Los rendimientos pasados no garantizan resultados futuros
> - Toda inversión conlleva riesgo de pérdida de capital
> - Consulte a un asesor financiero certificado antes de invertir
> 
> **Marco Regulatorio Colombia**: Las inversiones en valores están reguladas por la Superintendencia Financiera de Colombia bajo el Decreto 2555 de 2010 y la Ley 964 de 2005.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

<p align="center">
  <i>Desarrollado con ❤️ y Python</i>
</p>
