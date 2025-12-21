# 📊 Risk Management 2025 - Portfolio Construction via Clustering

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Objetivo

Construcción de carteras de inversión diversificadas utilizando técnicas de **clustering no supervisado** sobre más de 300 activos financieros (acciones y ETFs), con **backtesting** mensual comparado contra el benchmark **SPY (S&P 500)**.

## 📋 Metodología

1. **EDA**: Análisis exploratorio de datos de precios diarios (2010-2025)
2. **Feature Engineering**: Cálculo de métricas de riesgo y retorno
   - Retornos, Volatilidad, Sharpe Ratio, Sortino Ratio, Calmar Ratio
   - Beta vs SPY, Correlación con benchmark
   - VaR, CVaR, Maximum Drawdown
3. **Clustering**: Aplicación de 3 técnicas avanzadas
   - K-Means (con método del codo)
   - Hierarchical Clustering (dendrograma)
   - HDBSCAN (detección de outliers)
4. **Portfolio Construction**: Selección de representantes por cluster con reglas anti-concentración
5. **Backtesting**: Simulación 2015-2024 con rebalanceo mensual y costos de transacción

## 📁 Estructura del Proyecto

```
riskmanagement2025/
├── data/                    # Datos (no incluidos en repo)
│   └── trading_data.db      # Base de datos SQLite
├── notebooks/               # Jupyter notebooks numerados
│   ├── 01_eda_data_loading.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering_analysis.ipynb
│   ├── 04_portfolio_construction.ipynb
│   ├── 05_backtesting.ipynb
│   └── 06_results_report.ipynb
├── src/                     # Módulos de funciones reutilizables
│   ├── data_loader.py
│   ├── features.py
│   ├── clustering.py
│   ├── portfolio.py
│   └── backtesting.py
├── reports/                 # Resultados y gráficos
│   └── figures/
├── config/                  # Configuraciones
├── requirements.txt         # Dependencias
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

## 📈 Resultados

*(Se actualizará al completar el análisis)*

| Métrica | Portfolio | SPY (Benchmark) |
|---------|-----------|-----------------|
| Retorno Anual | - | - |
| Volatilidad | - | - |
| Sharpe Ratio | - | - |
| Max Drawdown | - | - |

## ⚠️ Limitaciones

- Los datos históricos **no garantizan rendimiento futuro**
- Costos de transacción estimados (0.1%), no incluye slippage
- No considera restricciones de liquidez ni market impact
- Rebalanceo mensual asume ejecución al cierre del día
- No incluye dividendos en el cálculo de retornos

## 🛠️ Tecnologías

- **Python 3.11+**
- **Pandas / NumPy**: Manipulación de datos
- **Scikit-learn / HDBSCAN**: Clustering
- **Matplotlib / Seaborn / Plotly**: Visualización
- **SQLite / SQLAlchemy**: Base de datos

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 👤 Autor

Juan Carlos Ruiz Arteaga, carlosarte11@gmail.com
---

*Proyecto desarrollado con fines educativos y de investigación. No constituye asesoría financiera.*
