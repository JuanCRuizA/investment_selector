# Portfolio Construction via Clustering -- Sistema de Gestion de Portafolios

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B.svg)](https://streamlit.io/cloud)
[![Status](https://img.shields.io/badge/Status-Production-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Pending_CI-yellow.svg)]()

Sistema cuantitativo de seleccion y construccion de portafolios de inversion basado en clustering K-Means, disenado para clasificar activos financieros segun caracteristicas cuantitativas y adaptar las recomendaciones al perfil de riesgo del inversionista.

**[Acceder a la Aplicacion](https://stocks-portfolio-selector.streamlit.app)** *(Streamlit Cloud)*

---

## Tabla de Contenidos

1. [Arquitectura de la Aplicacion](#1-arquitectura-de-la-aplicacion)
2. [Documentacion Tecnica](#2-documentacion-tecnica)
3. [Deuda Tecnica y CI/CD](#3-deuda-tecnica-y-cicd)
4. [Seguridad y Escalabilidad](#4-seguridad-y-escalabilidad)
5. [Machine Learning -- Modelos y Futuro](#5-machine-learning----modelos-y-futuro)
6. [Optimizacion y Rendimiento](#6-optimizacion-y-rendimiento)
7. [Conclusiones](#7-conclusiones)
8. [Backlog -- Hoja de Ruta](#8-backlog----hoja-de-ruta)
9. [Esquema de Paper Academico](#9-esquema-de-paper-academico)
10. [Instalacion y Uso](#10-instalacion-y-uso)
11. [Autor](#11-autor)
12. [Disclaimer](#12-disclaimer)

---

## 1. Arquitectura de la Aplicacion

### 1.1 Diagrama General del Sistema

```
+===========================================================================+
|                     ARQUITECTURA DEL SISTEMA                              |
+===========================================================================+
|                                                                           |
|  CAPA DE DATOS            CAPA DE PROCESAMIENTO         CAPA DE CONSUMO   |
|  +--------------+         +-------------------+         +---------------+ |
|  |              |         |                   |         |               | |
|  | SQLite DB    |-------->| Pipeline ML       |-------->| Streamlit App | |
|  | (1.6M rows)  |         | (5 etapas)        |         | (Dashboard)   | |
|  |              |         |                   |         |               | |
|  | prices_daily |         | 01_Ingestion      |         | Portafolio    | |
|  | 468 tickers  |         | 02_Features       |         | Backtesting   | |
|  | 2010-2025    |         | 03_Clustering     |         | Metricas      | |
|  |              |         | 04_Portfolios     |         | Comparacion   | |
|  |              |         | 05_Reports        |         | Exportacion   | |
|  +--------------+         +-------------------+         +---------------+ |
|        |                         |                            |           |
|        v                         v                            v           |
|  data/*.csv               reports/*.csv              outputs/api/*.csv    |
|  features_matrix.csv      backtest_*.csv             portfolios.csv       |
|  segmentacion_final/      portafolio_*.csv           equity_curves.csv    |
|                                                                           |
+===========================================================================+
```

### 1.2 Flujo de Datos del Pipeline

```
+----------------+     +-------------------+     +-----------------+
| STAGE 1        |     | STAGE 2           |     | STAGE 3         |
| Data Ingestion |---->| Feature           |---->| Clustering      |
|                |     | Engineering       |     |                 |
| - SQLite load  |     | - 21 metricas     |     | - DBSCAN        |
| - Validacion   |     | - Retorno, riesgo |     |   (outliers)    |
| - Train/Test   |     | - Momentum, beta  |     | - K-Means (K=4) |
| - 468 tickers  |     | - VaR, CVaR       |     | - PCA (2D)      |
+----------------+     +-------------------+     +-----------------+
                                                        |
                        +-------------------+           |
                        | STAGE 5           |           v
                        | Report Generation |     +-----------------+
                        |                   |<----| STAGE 4         |
                        | - Consolidar CSV  |     | Portfolio       |
                        | - API outputs     |     | Selection       |
                        | - Metadata JSON   |     |                 |
                        +-------------------+     | - Scoring       |
                              |                   | - 5 perfiles    |
                              v                   | - Backtesting   |
                        +-------------------+     +-----------------+
                        | STREAMLIT APP     |
                        | (Consumo final)   |
                        +-------------------+
```

### 1.3 Arquitectura de la Aplicacion Web (Streamlit)

```
streamlit_app/
|
+-- app.py                      # Punto de entrada principal
|   +-- init_data_loader()      # Cache de datos (@st.cache_resource)
|   +-- init_portfolio_selector()
|   +-- render_header()
|   +-- render_tabs()           # Orquestador de pestanas
|   +-- main()
|
+-- core/                       # Logica de negocio
|   +-- data_loader.py          # DataLoader: carga CSV con cache TTL 1h
|   +-- portfolio_selector.py   # Seleccion de portafolio por perfil
|   +-- calculations.py         # Calculos de metricas financieras
|
+-- components/                 # Componentes de UI
|   +-- sidebar.py              # Panel lateral: perfil, monto, horizonte
|   +-- portfolio_view.py       # Vista de composicion del portafolio
|   +-- backtest_view.py        # Curvas de equity y metricas de backtest
|   +-- metrics_view.py         # Metricas detalladas por activo
|   +-- comparison_view.py      # Comparacion entre perfiles
|   +-- export_utils.py         # Exportacion CSV/Excel/PDF
|
+-- utils/                      # Utilidades
    +-- charts.py               # ChartFactory: graficos Plotly
    +-- formatters.py           # Formateo de datos y paleta de colores
```

### 1.4 Patrones Arquitectonicos

| Patron | Implementacion | Ubicacion |
|--------|----------------|-----------|
| **Pipeline Pattern** | 5 etapas secuenciales con CLI | `pipeline/run_pipeline.py` |
| **Factory Pattern** | `ChartFactory` para graficos Plotly | `utils/charts.py` |
| **Singleton/Cache** | `@st.cache_resource` para DataLoader | `app.py` |
| **Strategy Pattern** | Perfiles de inversion intercambiables | `config/profiles.yaml` |
| **MVC** | Model (core/), View (components/), Controller (app.py) | `streamlit_app/` |
| **Configuration-Driven** | YAML centralizado para parametros | `config/` |

---

## 2. Documentacion Tecnica

### 2.1 Stack Tecnologico

| Componente | Tecnologia | Version | Proposito |
|------------|-----------|---------|-----------|
| Lenguaje | Python | 3.11+ | Core de desarrollo |
| Framework Web | Streamlit | 1.28+ | Dashboard interactivo |
| ML/Clustering | scikit-learn | 1.3+ | K-Means, PCA, StandardScaler |
| Outlier Detection | hdbscan | 0.8.29+ | DBSCAN para anomalias |
| Datos | pandas / numpy | 2.0+ / 1.24+ | Manipulacion de datos |
| Visualizacion | Plotly | 5.14+ | Graficos interactivos |
| Estadisticas | scipy / empyrical | 1.10+ / 0.5+ | Metricas financieras |
| Base de datos | SQLite / SQLAlchemy | 2.0+ | Almacenamiento de precios |
| Configuracion | PyYAML | 6.0+ | Parametros del pipeline |
| Exportacion | fpdf2 / openpyxl | -- | PDF y Excel |

### 2.2 Estructura del Proyecto

```
riskmanagement2025/
|
+-- data/                               # Datos de entrada y procesados
|   +-- trading_data.db                 # Base de datos SQLite (1.6M registros)
|   +-- prices_train.csv                # Precios historicos 2010-2023
|   +-- prices_test.csv                 # Precios de prueba 2024-2025
|   +-- features_matrix.csv            # 21 metricas x 467 activos
|   +-- segmentacion_final/            # Resultados de clustering
|
+-- config/                             # Configuracion centralizada
|   +-- settings.yaml                   # Parametros del pipeline (176 lineas)
|   +-- profiles.yaml                   # Definicion de 5 perfiles de inversion
|
+-- src/                                # Modulos core
|   +-- data_loader.py                  # Carga de datos desde SQLite
|   +-- features.py                     # Calculo de 21 metricas financieras
|   +-- clustering.py                   # DBSCAN + K-Means + PCA
|   +-- portfolio.py                    # Construccion y scoring de portafolios
|   +-- backtesting.py                  # Motor de backtesting
|   +-- utils.py                        # Utilidades (logging, config, paths)
|
+-- pipeline/                           # Pipeline de produccion
|   +-- run_pipeline.py                 # Orquestador CLI principal
|   +-- 01_data_ingestion.py            # Etapa 1: Ingestion de datos
|   +-- 02_feature_engineering.py       # Etapa 2: Feature Engineering
|   +-- 03_clustering.py               # Etapa 3: Segmentacion
|   +-- 04_portfolio_selection.py       # Etapa 4: Portafolios + Backtest
|   +-- 05_generate_reports.py          # Etapa 5: Generacion de reportes
|
+-- streamlit_app/                      # Aplicacion web
|   +-- app.py                          # Punto de entrada
|   +-- core/                           # Logica de negocio
|   +-- components/                     # Componentes de UI
|   +-- utils/                          # Graficos y formateo
|   +-- requirements.txt                # Dependencias de la app
|
+-- reports/                            # Resultados de backtesting
+-- outputs/api/                        # Archivos para consumo web
+-- notebooks/                          # Notebooks de desarrollo
+-- logs/                               # Logs del pipeline
+-- requirements.txt                    # Dependencias del proyecto
+-- LICENSE                             # Licencia MIT
```

### 2.3 Modulos Core (src/)

#### data_loader.py
- **Funciones principales**: `connect_database()`, `load_prices()`, `get_valid_tickers()`, `pivot_prices()`, `split_train_test()`, `fill_missing_prices()`, `run_data_ingestion()`
- **Responsabilidad**: Conexion a SQLite, validacion de datos, transformacion long-to-wide, division train/test
- **Filtros**: Minimo 5 anos de historial (1,260 observaciones), imputacion de `adj_close` desde `close`

#### features.py
- **Funciones principales**: `calculate_returns()`, `calculate_volatility()`, `calculate_sharpe()`, `calculate_beta()`, `calculate_max_drawdown()`, `calculate_alpha()`, `calculate_all_features_extended()`
- **Metricas calculadas** (21 total):
  - **Retornos** (3): Total, anualizado, promedio diario
  - **Riesgo** (5): Volatilidad, desviacion a la baja, max drawdown, VaR 95%, CVaR 95%
  - **Ratios de eficiencia** (3): Sharpe, Sortino, Calmar
  - **Exposicion de mercado** (4): Beta, Alpha, R-cuadrado, correlacion con SPY
  - **Distribucion** (3): Asimetria, curtosis, ratio de retornos positivos
  - **Momentum/Volatilidad** (3): Momentum 6M, volatilidad de la volatilidad, ratio ganancia/perdida

#### clustering.py
- **Funciones principales**: `prepare_features()`, `find_optimal_k()`, `apply_kmeans()`, `detect_outliers_dbscan()`, `run_hybrid_clustering()`
- **Algoritmo hibrido**:
  1. DBSCAN para deteccion de outliers (eps calculado desde percentil 90 de distancias k-NN)
  2. K-Means (K=4) sobre los activos no-outlier
  3. PCA para reduccion a 2 dimensiones (visualizacion)

#### portfolio.py
- **Formula de scoring compuesto**:
  ```
  Score = 0.35 x Return_norm + 0.30 x Momentum_6m_norm + 0.15 x Sharpe_norm + 0.20 x Beta_adj
  ```
- **Reglas de concentracion**: Maximo 20% por activo, maximo 40% por cluster, 10 activos por portafolio

#### backtesting.py
- **Estrategia**: Buy & Hold con rebalanceo mensual
- **Parametros**: Capital inicial $10,000, costos de transaccion 0.1% (10 bps round-trip)
- **Benchmark**: SPY (S&P 500 ETF)

### 2.4 Segmentacion de Activos (Resultados)

| Cluster | Nombre | Caracteristicas | Activos | % del Total |
|---------|--------|-----------------|---------|-------------|
| -1 | Outliers | Rendimientos extremos, alta volatilidad | 34 | 7.3% |
| 0 | Conservador | Baja volatilidad, beta bajo, retornos estables | 151 | 32.3% |
| 1 | Alto Rendimiento | Retornos superiores, momentum fuerte, beta elevado | 161 | 34.5% |
| 2 | Moderado | Balance riesgo-retorno equilibrado | 116 | 24.8% |
| 3 | Estable | Volatilidad muy baja, beta cercano a cero | 5 | 1.1% |

**Total de activos analizados**: 467 (S&P 500 + ETFs principales)

### 2.5 Perfiles de Inversion

| Perfil | Distribucion por Cluster | Descripcion |
|--------|--------------------------|-------------|
| **Conservador** | 60% Estable + 20% Conservador + 20% Moderado | Preservacion de capital |
| **Moderado** | 40% Alto Rend. + 30% Moderado + 30% Estable | Balance crecimiento/estabilidad |
| **Normal** | 20% cada cluster (5 clusters) | Maxima diversificacion |
| **Agresivo** | 70% Alto Rend. + 20% Moderado + 10% Outliers | Crecimiento agresivo |
| **Especulativo** | 50% Alto Rend. + 30% Outliers + 20% Moderado | Maximo riesgo/retorno |

### 2.6 Resultados de Backtesting (Out-of-Sample 2024-2025)

| Perfil | Retorno Total | Retorno Anualizado | Volatilidad | Sharpe | Max Drawdown | Sortino | Calmar |
|--------|---------------|--------------------|-----------:|-------:|-------------:|--------:|-------:|
| Conservador | 27.64% | 13.23% | 16.90% | 0.52 | -17.97% | 0.72 | 0.74 |
| Moderado | 13.02% | 6.43% | 13.25% | 0.15 | -17.31% | 0.21 | 0.37 |
| Agresivo | 27.19% | 13.03% | 17.18% | 0.50 | -21.54% | 0.70 | 0.60 |
| Especulativo | 50.16% | 23.00% | 21.96% | 0.84 | -26.22% | 1.17 | 0.88 |
| Normal | 61.86% | 27.78% | 24.66% | 0.94 | -24.99% | 1.27 | 1.11 |
| **SPY (Benchmark)** | **43.84%** | **20.33%** | **16.47%** | **0.96** | **-19.00%** | **1.23** | **1.07** |

**Hallazgos clave**:
- Los perfiles Normal y Especulativo superaron al SPY en retorno total
- El perfil Conservador ofrecio mejor proteccion ante drawdowns con retorno aceptable
- El perfil Normal obtuvo el mejor Calmar ratio (1.11), superando al benchmark

---

## 3. Deuda Tecnica y CI/CD

### 3.1 Deuda Tecnica Identificada

| Prioridad | Area | Descripcion | Impacto |
|-----------|------|-------------|---------|
| **ALTA** | Testing | No existen tests unitarios ni de integracion | Riesgo de regresiones |
| **ALTA** | CI/CD | No hay pipeline de integracion continua | Deploys sin validacion |
| **ALTA** | Validacion de datos | Sin validacion de inputs del usuario en la app web | Riesgo de errores en runtime |
| **MEDIA** | Type Hints | Cobertura parcial de type hints | Dificultad de mantenimiento |
| **MEDIA** | Error Handling | Manejo de errores generico (`except Exception`) | Errores silenciosos |
| **MEDIA** | Documentacion de API | Sin docstrings en algunas funciones publicas | Dificultad de onboarding |
| **BAJA** | Dependencias | Algunas dependencias sin version fija | Posibles incompatibilidades |
| **BAJA** | Logs | Logging basico, sin metricas de rendimiento | Dificultad de diagnostico |

### 3.2 Plan de Testing Propuesto

```
tests/
|
+-- unit/
|   +-- test_data_loader.py         # Tests de carga y validacion de datos
|   +-- test_features.py            # Tests de calculo de 21 metricas
|   +-- test_clustering.py          # Tests de K-Means y DBSCAN
|   +-- test_portfolio.py           # Tests de scoring y seleccion
|   +-- test_backtesting.py         # Tests del motor de backtesting
|   +-- test_utils.py               # Tests de utilidades
|
+-- integration/
|   +-- test_pipeline.py            # Test end-to-end del pipeline
|   +-- test_data_integrity.py      # Validacion de integridad de datos
|   +-- test_streamlit_app.py       # Tests de la aplicacion web
|
+-- fixtures/
|   +-- sample_prices.csv           # Datos de prueba reducidos
|   +-- expected_features.csv       # Resultados esperados de features
|   +-- expected_clusters.csv       # Resultados esperados de clustering
|
+-- conftest.py                     # Configuracion de pytest y fixtures
```

### 3.3 Pipeline de CI/CD Propuesto (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ruff mypy
      - run: ruff check src/ pipeline/ streamlit_app/
      - run: mypy src/ --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest tests/ --cov=src --cov-report=xml -v
      - uses: codecov/codecov-action@v3

  deploy:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy notification
        run: echo "Streamlit Cloud auto-deploys from main branch"
```

### 3.4 Herramientas de Calidad Recomendadas

| Herramienta | Proposito | Configuracion |
|-------------|-----------|---------------|
| **pytest** | Framework de testing | `pytest.ini` o `pyproject.toml` |
| **pytest-cov** | Cobertura de codigo | Target: 80% minimo |
| **ruff** | Linter y formatter (reemplaza flake8+black) | `ruff.toml` |
| **mypy** | Verificacion de tipos estaticos | `mypy.ini` |
| **pre-commit** | Hooks de pre-commit | `.pre-commit-config.yaml` |
| **Codecov** | Reporte de cobertura | Integracion con GitHub |

---

## 4. Seguridad y Escalabilidad

### 4.1 Estado Actual de Seguridad

| Aspecto | Estado Actual | Riesgo |
|---------|---------------|--------|
| Autenticacion | Sin autenticacion | ALTO -- cualquier persona puede acceder |
| Autorizacion | Sin roles de usuario | ALTO -- sin control de acceso |
| Datos sensibles | No aplica (datos publicos de mercado) | BAJO |
| Inyeccion de codigo | Streamlit protege contra XSS basico | MEDIO |
| Base de datos | SQLite local (solo lectura) | BAJO |
| HTTPS | Streamlit Cloud provee SSL automatico | OK |
| Variables de entorno | Sin secretos expuestos | OK |

### 4.2 Administracion de Usuarios (Propuesta a Futuro)

Para escalar a una aplicacion con gestion de usuarios, se recomienda implementar:

**Requerimientos**:
- Registro e inicio de sesion con usuario y contrasena
- Roles: Administrador, Analista, Inversionista (solo lectura)
- Historial de portafolios generados por usuario
- Preferencias de perfil de riesgo persistentes

**Base de datos recomendada**: PostgreSQL (robusta, escalable, gratuita en tier basico de nube)

### 4.3 Arquitectura de Escalamiento Propuesta

Se propone una arquitectura de 3 capas, economica y profesional:

```
+===========================================================================+
|                  ARQUITECTURA DE ESCALAMIENTO                             |
+===========================================================================+
|                                                                           |
|  OPCION A: Monolito Moderno (Recomendada para MVP escalado)               |
|  Costo estimado: $20-50 USD/mes                                          |
|                                                                           |
|  +-------------------+    +-------------------+    +------------------+   |
|  | FRONTEND          |    | BACKEND           |    | BASE DE DATOS    |   |
|  |                   |    |                   |    |                  |   |
|  | Next.js / React   |<-->| FastAPI (Python)  |<-->| PostgreSQL       |   |
|  | - Dashboard       |    | - REST API        |    | - Usuarios       |   |
|  | - Auth UI         |    | - Auth (JWT)      |    | - Portafolios    |   |
|  | - Graficos        |    | - ML Pipeline     |    | - Historico      |   |
|  | - Responsive      |    | - WebSockets      |    | - Configuracion  |   |
|  |                   |    |                   |    |                  |   |
|  | Vercel (gratis)   |    | Railway / Render  |    | Supabase (free)  |   |
|  +-------------------+    +-------------------+    +------------------+   |
|                                                                           |
+===========================================================================+
|                                                                           |
|  OPCION B: Microservicios en la Nube (Escala empresarial)                 |
|  Costo estimado: $100-300 USD/mes                                        |
|                                                                           |
|  +-------+    +----------+    +---------+    +----------+    +--------+  |
|  | CDN   |    | API      |    | ML      |    | Database |    | Cache  |  |
|  | Cloud |    | Gateway  |    | Service |    | Service  |    | Layer  |  |
|  | Front |    |          |    |         |    |          |    |        |  |
|  | (S3+  |--->| Azure    |--->| Azure   |--->| Azure    |--->| Redis  |  |
|  |  CF)  |    | API Mgmt |    | Funct.  |    | DB for   |    | Cache  |  |
|  |       |    | / AWS    |    | / AWS   |    | Postgres |    |        |  |
|  |       |    | API GW   |    | Lambda  |    | / RDS    |    |        |  |
|  +-------+    +----------+    +---------+    +----------+    +--------+  |
|                                                                           |
+===========================================================================+
```

### 4.4 Comparativa de Opciones de Nube

| Servicio | AWS | Azure | GCP | Recomendacion |
|----------|-----|-------|-----|---------------|
| **Compute** | EC2 / Lambda | App Service / Functions | Cloud Run | Azure App Service (tier F1 gratis) |
| **Database** | RDS PostgreSQL | Azure DB for PostgreSQL | Cloud SQL | Supabase (tier gratis con 500MB) |
| **Auth** | Cognito | Azure AD B2C | Firebase Auth | Supabase Auth (integrado, gratis) |
| **Storage** | S3 | Blob Storage | Cloud Storage | S3 / Vercel (tier gratis) |
| **CI/CD** | CodePipeline | Azure DevOps | Cloud Build | GitHub Actions (gratis) |
| **Costo mensual** | ~$50-100 | ~$30-80 | ~$40-90 | **$20-50 con tiers gratuitos** |

### 4.5 Recomendacion de Arquitectura Economica y Profesional

**Stack recomendado para produccion**:

| Capa | Tecnologia | Costo | Justificacion |
|------|-----------|-------|---------------|
| **Frontend** | Next.js + React | $0 (Vercel free) | SSR, SEO, rendimiento, ecosistema |
| **Backend API** | FastAPI (Python) | $7/mes (Railway) | Reutiliza codigo Python existente, async nativo, OpenAPI auto |
| **Base de datos** | PostgreSQL via Supabase | $0 (free tier) | Auth integrada, API REST auto, 500MB gratis |
| **Cache** | Redis via Upstash | $0 (free tier) | 10K comandos/dia gratis, reduce carga DB |
| **ML Pipeline** | GitHub Actions (scheduled) | $0 (free tier) | 2,000 min/mes gratis, cron jobs |
| **Monitoreo** | Sentry + Vercel Analytics | $0 (free tier) | Errores + rendimiento |
| **Total** | -- | **~$7-15/mes** | Profesional y escalable |

**Ventajas de esta arquitectura**:
- **FastAPI** reutiliza todo el codigo Python/ML existente sin reescritura
- **Supabase** provee PostgreSQL + Auth + API REST en un solo servicio
- **Next.js** permite SSR para rendimiento y SEO
- **Escalamiento horizontal** cuando sea necesario sin cambio de arquitectura

### 4.6 Esquema de Base de Datos Propuesto

```sql
-- Tabla de usuarios
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'investor',  -- admin, analyst, investor
    risk_profile VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Tabla de portafolios generados
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    profile_name VARCHAR(50) NOT NULL,
    investment_amount DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Tabla de activos del portafolio
CREATE TABLE portfolio_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES portfolios(id),
    ticker VARCHAR(10) NOT NULL,
    weight DECIMAL(5,4),
    segment INTEGER,
    score DECIMAL(10,6)
);

-- Tabla de sesiones de backtest
CREATE TABLE backtest_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES portfolios(id),
    total_return DECIMAL(10,6),
    annualized_return DECIMAL(10,6),
    volatility DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    benchmark_return DECIMAL(10,6),
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 5. Machine Learning -- Modelos y Futuro

### 5.1 Modelos Actuales

| Modelo | Tipo | Uso en el Sistema | Parametros |
|--------|------|-------------------|------------|
| **K-Means** | Clustering no supervisado | Segmentar 467 activos en 4 clusters | K=4, random_state=42, n_init=10 |
| **DBSCAN** | Clustering basado en densidad | Deteccion de outliers (34 activos atipicos) | eps=percentil 90, min_samples=5 |
| **StandardScaler** | Preprocesamiento | Normalizacion de features (media=0, std=1) | Fit en datos de entrenamiento |
| **PCA** | Reduccion de dimensionalidad | Visualizacion 2D de clusters | n_components=2 |
| **Min-Max Scoring** | Ranking | Puntuar activos dentro de cada segmento | Pesos: 35/30/15/20 |

### 5.2 Modelos para Mejora Futura

| Modelo | Tipo | Aplicacion Propuesta | Beneficio Esperado |
|--------|------|----------------------|---------------------|
| **Gaussian Mixture Models (GMM)** | Clustering probabilistico | Asignacion blanda de activos a clusters | Mejor manejo de activos en fronteras |
| **Random Forest / XGBoost** | Ensemble supervisado | Prediccion de retornos futuros (clasificacion) | Incorporar senales predictivas al scoring |
| **LSTM / GRU** | Deep Learning secuencial | Prediccion de series de tiempo de precios | Capturar patrones temporales complejos |
| **Autoencoders** | Deep Learning no supervisado | Reduccion de dimensionalidad no lineal | Mejores representaciones latentes que PCA |
| **Hierarchical Risk Parity (HRP)** | Optimizacion de portafolios | Reemplazar equal-weight por ponderacion optima | Mejor diversificacion basada en correlaciones |
| **Black-Litterman** | Optimizacion bayesiana | Incorporar vistas de mercado al portafolio | Combinar datos cuantitativos con opiniones |
| **Reinforcement Learning (DQN/PPO)** | Aprendizaje por refuerzo | Rebalanceo dinamico adaptativo | Optimizacion de decisiones secuenciales |

### 5.3 Integracion de Agentes de IA Generativa

Para evolucion futura, se propone un sistema multi-agente que combine analisis cuantitativo con recomendaciones personalizadas:

```
+===========================================================================+
|            SISTEMA MULTI-AGENTE PARA ASESORIA DE INVERSIONES              |
+===========================================================================+
|                                                                           |
|  +------------------+    +------------------+    +-------------------+    |
|  | AGENTE DE        |    | AGENTE DE        |    | AGENTE DE         |    |
|  | ANALISIS         |    | MERCADO          |    | PERFIL DE         |    |
|  | CUANTITATIVO     |    |                  |    | INVERSOR          |    |
|  |                  |    | - Noticias       |    |                   |    |
|  | - Features       |    | - Sentimiento    |    | - Objetivos       |    |
|  | - Clustering     |    | - Indicadores    |    | - Tolerancia      |    |
|  | - Backtesting    |    |   macroeconomicos|    | - Horizonte       |    |
|  | - Scoring        |    | - Eventos corp.  |    | - Restricciones   |    |
|  +--------+---------+    +--------+---------+    +---------+---------+    |
|           |                       |                        |              |
|           +----------+------------+------------+-----------+              |
|                      |                         |                          |
|                      v                         v                          |
|            +-------------------+     +-------------------+               |
|            | AGENTE            |     | AGENTE            |               |
|            | ORQUESTADOR       |     | DE RIESGO         |               |
|            |                   |     |                   |               |
|            | - Combina inputs  |     | - VaR/CVaR        |               |
|            | - Genera          |     | - Stress testing  |               |
|            |   recomendacion   |     | - Correlaciones   |               |
|            | - Formato natural |     | - Limites         |               |
|            +--------+----------+     +-------------------+               |
|                     |                                                     |
|                     v                                                     |
|            +-------------------+                                          |
|            | RECOMENDACION     |                                          |
|            | PERSONALIZADA     |                                          |
|            |                   |                                          |
|            | "Dado su perfil   |                                          |
|            |  moderado y un    |                                          |
|            |  horizonte de 5   |                                          |
|            |  anios, se sugiere|                                          |
|            |  el siguiente     |                                          |
|            |  portafolio..."   |                                          |
|            +-------------------+                                          |
|                                                                           |
+===========================================================================+
```

**Implementacion tecnica sugerida**:

| Componente | Tecnologia | Funcion |
|------------|-----------|---------|
| LLM Base | Claude API / GPT-4 | Generacion de recomendaciones en lenguaje natural |
| Framework de agentes | LangChain / CrewAI | Orquestacion de agentes especializados |
| RAG (Retrieval) | ChromaDB / Pinecone | Contexto de informes financieros y reportes |
| Datos en tiempo real | Alpha Vantage / Yahoo Finance API | Precios actualizados y noticias |
| Analisis de sentimiento | FinBERT | Sentimiento de noticias financieras |
| Voz del experto | Prompt Engineering + Few-Shot | Personalidad de analista financiero senior |

**Roles de los agentes**:

1. **Agente de Analisis Cuantitativo**: Ejecuta el pipeline ML existente, calcula metricas, genera scoring
2. **Agente de Mercado**: Analiza condiciones macroeconomicas, sentimiento, noticias relevantes
3. **Agente de Perfil de Inversor**: Interpreta preferencias, restricciones y objetivos del usuario
4. **Agente de Riesgo**: Evalua riesgos del portafolio propuesto, realiza stress testing
5. **Agente Orquestador**: Combina todos los inputs y genera una recomendacion personalizada en lenguaje natural

---

## 6. Optimizacion y Rendimiento

### 6.1 Metricas Actuales de Rendimiento

| Etapa del Pipeline | Duracion | Cuello de Botella |
|--------------------|----------|-------------------|
| Data Ingestion | ~13s | Query SQLite y pivot |
| Feature Engineering | ~7s | 21 metricas x 467 activos |
| Clustering | ~2s | Convergencia K-Means |
| Portfolio Selection | ~1s | Scoring y seleccion |
| Report Generation | ~0.3s | Escritura de archivos |
| **Total Pipeline** | **~23s** | Data Ingestion |

### 6.2 Recomendaciones de Optimizacion

#### Rendimiento del Pipeline

| Recomendacion | Impacto | Complejidad | Detalle |
|---------------|---------|-------------|---------|
| **Paralelizar Feature Engineering** | ALTO | Media | Usar `joblib.Parallel` o `concurrent.futures` para calcular metricas por activo en paralelo |
| **Cache de features** | ALTO | Baja | Implementar cache con hash de datos de entrada; si los datos no cambian, reutilizar features |
| **Reemplazar SQLite por Parquet** | MEDIO | Baja | Parquet es 5-10x mas rapido para lectura columnar; ideal para datos de series de tiempo |
| **Incremental backtest** | MEDIO | Media | Solo recalcular backtest para periodos nuevos en vez de recalcular todo |
| **Vectorizacion con NumPy** | MEDIO | Baja | Reemplazar loops de Python por operaciones vectorizadas donde aplique |

#### Rendimiento de la Aplicacion Web

| Recomendacion | Impacto | Complejidad | Detalle |
|---------------|---------|-------------|---------|
| **Pre-calcular todos los graficos** | ALTO | Media | Generar graficos como HTML estatico en el pipeline y servirlos directamente |
| **Optimizar cache TTL** | MEDIO | Baja | Ajustar `@st.cache_data(ttl=)` segun frecuencia de actualizacion de datos |
| **Lazy loading de tabs** | MEDIO | Baja | Cargar datos de cada tab solo cuando el usuario la selecciona |
| **Compresion de CSV** | BAJO | Baja | Servir CSV comprimidos (gzip) para reducir tiempo de carga |
| **CDN para assets estaticos** | MEDIO | Media | Servir imagenes y archivos estaticos desde CDN |
| **WebSocket para actualizacion** | BAJO | Alta | Actualizacion en tiempo real sin reload de pagina |

#### Rendimiento del Machine Learning

| Recomendacion | Impacto | Complejidad | Detalle |
|---------------|---------|-------------|---------|
| **Mini-Batch K-Means** | MEDIO | Baja | Para datasets mayores (>10K activos), usar MiniBatchKMeans |
| **GPU para clustering** | BAJO | Alta | cuML (RAPIDS) para K-Means acelerado por GPU |
| **Feature selection automatica** | MEDIO | Media | Usar Mutual Information o Boruta para seleccionar features relevantes |
| **Ensemble de clusterings** | MEDIO | Media | Combinar K-Means + GMM + Spectral para clusters mas robustos |

---

## 7. Conclusiones

### 7.1 Conclusiones Tecnicas

1. **El enfoque hibrido DBSCAN + K-Means es efectivo** para segmentar activos financieros. La deteccion previa de outliers mejora la calidad de los clusters principales al evitar que activos atipicos distorsionen los centroides.

2. **21 metricas financieras proporcionan una caracterizacion robusta** de los activos. La seleccion de 10 features para clustering (retorno, volatilidad, Sharpe, Sortino, max drawdown, VaR, CVaR, beta, asimetria, curtosis) captura las dimensiones fundamentales de riesgo-retorno.

3. **La arquitectura de pipeline modular (5 etapas) facilita el mantenimiento** y permite reentrenar componentes individuales sin ejecutar todo el sistema. La configuracion centralizada en YAML elimina hardcoding.

4. **La deuda tecnica principal es la ausencia de tests automatizados**. Sin cobertura de tests, cada cambio en el pipeline representa un riesgo de regresion no detectado.

5. **Streamlit es adecuado para la fase actual (MVP/prototipo)** pero presenta limitaciones para escalar a produccion con multiples usuarios concurrentes.

### 7.2 Conclusiones de Negocio

1. **Los portafolios generados por el sistema son competitivos con el benchmark (SPY)**. En el periodo out-of-sample 2024-2025, los perfiles Normal (+61.86%) y Especulativo (+50.16%) superaron al SPY (+43.84%).

2. **El sistema cumple su objetivo de personalizar por perfil de riesgo**. Existe una correlacion clara entre el nivel de riesgo del perfil y la volatilidad/drawdown del portafolio resultante.

3. **El rebalanceo mensual y la diversificacion por clusters** son mecanismos efectivos para gestionar riesgo. El perfil Conservador logro un drawdown maximo de -17.97% vs -19.00% del SPY.

4. **La aplicacion web proporciona transparencia total** al inversionista: puede ver la composicion exacta, las metricas historicas y comparar perfiles antes de tomar una decision.

5. **El modelo de negocio puede evolucionar** hacia un servicio de asesoria cuantitativa (robo-advisor) con suscripcion mensual, agregando autenticacion y personalizacion avanzada.

### 7.3 Conclusiones de Arquitectura

1. **La separacion pipeline/aplicacion web es una decision correcta**. El pipeline genera artefactos estaticos (CSV) que la web consume, eliminando dependencia en tiempo real del procesamiento ML.

2. **La migracion a FastAPI + PostgreSQL + Next.js** es el siguiente paso natural para escalar a una aplicacion multi-usuario profesional, reutilizando el 100% del codigo Python/ML existente.

3. **La configuracion YAML centralizada** permite parametrizar el sistema sin cambiar codigo, facilitando experimentacion con diferentes configuraciones de clustering y scoring.

4. **El patron de 5 perfiles predefinidos es extensible**. Se pueden agregar perfiles personalizados modificando unicamente `profiles.yaml` sin tocar codigo.

---

## 8. Backlog -- Hoja de Ruta

### 8.1 Corto Plazo (1-3 meses)

| ID | Tarea | Prioridad | Estimacion | Dependencia |
|----|-------|-----------|------------|-------------|
| B-01 | Implementar tests unitarios para `src/` (pytest) | CRITICA | 2 semanas | -- |
| B-02 | Configurar GitHub Actions CI (lint + test) | CRITICA | 3 dias | B-01 |
| B-03 | Agregar validacion de inputs en la app web | ALTA | 1 semana | -- |
| B-04 | Implementar pre-commit hooks (ruff, mypy) | ALTA | 2 dias | -- |
| B-05 | Agregar cobertura de tests (target 80%) | ALTA | 2 semanas | B-01 |
| B-06 | Migrar datos de SQLite a Parquet para rendimiento | MEDIA | 3 dias | -- |
| B-07 | Paralelizar Feature Engineering con joblib | MEDIA | 3 dias | -- |
| B-08 | Documentar API interna con docstrings completos | MEDIA | 1 semana | -- |

### 8.2 Mediano Plazo (3-6 meses)

| ID | Tarea | Prioridad | Estimacion | Dependencia |
|----|-------|-----------|------------|-------------|
| B-09 | Disenar e implementar API REST con FastAPI | ALTA | 3 semanas | B-01, B-02 |
| B-10 | Implementar autenticacion JWT con Supabase | ALTA | 2 semanas | B-09 |
| B-11 | Crear frontend con Next.js + React | ALTA | 4 semanas | B-09 |
| B-12 | Migrar base de datos a PostgreSQL | ALTA | 1 semana | B-09 |
| B-13 | Implementar Hierarchical Risk Parity (HRP) | MEDIA | 2 semanas | -- |
| B-14 | Agregar modelos predictivos (XGBoost) al scoring | MEDIA | 3 semanas | B-13 |
| B-15 | Implementar actualizacion automatica de precios | MEDIA | 2 semanas | B-09 |
| B-16 | Agregar analisis de sentimiento con FinBERT | BAJA | 3 semanas | B-14 |
| B-17 | Implementar sistema multi-agente con LangChain | BAJA | 4 semanas | B-14, B-16 |
| B-18 | Agregar factores ESG al analisis | BAJA | 2 semanas | B-14 |

### 8.3 Largo Plazo (6-12 meses)

| ID | Tarea | Prioridad | Estimacion |
|----|-------|-----------|------------|
| B-19 | Implementar Reinforcement Learning para rebalanceo | MEDIA | 6 semanas |
| B-20 | Agregar cobertura con opciones (hedging) | MEDIA | 4 semanas |
| B-21 | Desplegar en infraestructura de nube (Azure/AWS) | ALTA | 2 semanas |
| B-22 | Implementar monitoreo y alertas (Sentry, Datadog) | MEDIA | 1 semana |
| B-23 | Certificacion de seguridad (OWASP, SOC2 basico) | MEDIA | 4 semanas |
| B-24 | App movil (React Native) | BAJA | 8 semanas |

---

## 9. Esquema de Paper Academico

### Titulo Propuesto

**"Construccion de Portafolios de Inversion Mediante Clustering Hibrido y Scoring Cuantitativo: Un Enfoque Basado en Machine Learning para la Gestion de Riesgo"**

### Estructura del Paper

```
1. RESUMEN / ABSTRACT
   - Objetivo: Proponer un sistema de construccion de portafolios basado en
     clustering hibrido (DBSCAN + K-Means) y scoring cuantitativo
   - Metodo: Pipeline de 5 etapas con 21 metricas financieras
   - Resultados: Portafolios competitivos vs SPY en periodo out-of-sample
   - Contribucion: Framework reproducible para asesoria cuantitativa

2. INTRODUCCION
   2.1 Contexto y motivacion
       - Crecimiento de robo-advisors y asesoria cuantitativa
       - Limitaciones de optimizacion clasica de Markowitz
       - Oportunidad de ML para segmentacion de activos
   2.2 Problema de investigacion
       - Como construir portafolios personalizados por perfil de riesgo
         usando tecnicas de clustering no supervisado
   2.3 Objetivos
       - General: Disenar un sistema de construccion de portafolios basado en ML
       - Especificos: Segmentar activos, definir perfiles, evaluar rendimiento
   2.4 Contribuciones del paper
   2.5 Estructura del documento

3. REVISION DE LITERATURA
   3.1 Teoria moderna de portafolios (Markowitz, 1952)
   3.2 Modelos de factores (Fama-French, Carhart)
   3.3 Clustering en finanzas
       - K-Means para agrupacion de activos (De Prado, 2016)
       - DBSCAN para deteccion de anomalias financieras
       - Hierarchical Risk Parity (Lopez de Prado, 2016)
   3.4 Machine Learning en gestion de portafolios
       - Ensemble methods para prediccion de retornos
       - Deep Learning para series de tiempo financieras
       - Reinforcement Learning para rebalanceo
   3.5 Robo-advisors y perfilamiento de inversionistas
   3.6 Gaps en la literatura

4. METODOLOGIA
   4.1 Datos
       - Universo: S&P 500 + ETFs (468 activos, 2010-2025)
       - Division temporal: Train (2010-2023) / Test (2024-2025)
       - Fuente y preprocesamiento
   4.2 Feature Engineering (21 metricas)
       - Metricas de retorno, riesgo, ratios, mercado, distribucion, momentum
       - Justificacion de cada metrica seleccionada
   4.3 Algoritmo de clustering hibrido
       4.3.1 Deteccion de outliers con DBSCAN
       4.3.2 Segmentacion principal con K-Means
       4.3.3 Seleccion de K optimo (metodo del codo + silhouette)
       4.3.4 Reduccion dimensional con PCA
   4.4 Formula de scoring compuesto
       - Componentes y pesos
       - Normalizacion Min-Max por segmento
   4.5 Construccion de portafolios por perfil de riesgo
       - 5 perfiles: Conservador, Moderado, Normal, Agresivo, Especulativo
       - Reglas de asignacion y concentracion
   4.6 Backtesting
       - Metodologia: Buy & Hold con rebalanceo mensual
       - Metricas de evaluacion: Retorno, Sharpe, Sortino, Max Drawdown, Calmar
       - Benchmark: SPY (S&P 500)

5. RESULTADOS
   5.1 Analisis de clusters
       - Estadisticas descriptivas por segmento
       - Visualizacion PCA y distribucion de activos
   5.2 Rendimiento de portafolios (out-of-sample)
       - Metricas por perfil vs benchmark
       - Curvas de equity y drawdown
       - Analisis de retornos mensuales
   5.3 Analisis de sensibilidad
       - Variacion de K en K-Means
       - Variacion de pesos en formula de scoring
       - Impacto de costos de transaccion

6. DISCUSION
   6.1 Interpretacion de resultados
   6.2 Comparacion con literatura existente
   6.3 Ventajas del enfoque hibrido vs metodos tradicionales
   6.4 Limitaciones del estudio
       - Sesgo de supervivencia
       - Periodo de backtesting limitado
       - Ausencia de costos friccionales completos
   6.5 Implicaciones practicas

7. CONCLUSIONES Y TRABAJO FUTURO
   7.1 Conclusiones principales
   7.2 Trabajo futuro
       - Incorporacion de modelos predictivos (XGBoost, LSTM)
       - Optimizacion de portafolios con HRP y Black-Litterman
       - Integracion de analisis de sentimiento
       - Agentes de IA para asesoria personalizada
       - Reinforcement Learning para rebalanceo dinamico

8. REFERENCIAS BIBLIOGRAFICAS
   - Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
   - Fama, E. & French, K. (1993). Common Risk Factors. Journal of Financial Economics.
   - Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform OOS.
   - Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
   - Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via ML. RFS.
   - Zhang, Z., Zohren, S., & Roberts, S. (2020). Deep RL for Portfolio Management.
   - Ester, M. et al. (1996). A Density-Based Algorithm (DBSCAN). KDD.

9. ANEXOS
   A. Lista completa de 21 metricas y formulas
   B. Composicion detallada de portafolios por perfil
   C. Codigo fuente del pipeline (referencia a repositorio)
   D. Tablas de backtest completas
```

---

## 10. Instalacion y Uso

### Prerrequisitos
- Python 3.11+
- Conda (recomendado) o pip

### Instalacion

```bash
# 1. Clonar el repositorio
git clone https://github.com/fantastic1121/stocks_portfolio_selector.git
cd stocks_portfolio_selector

# 2. Crear ambiente virtual
conda create -n riskmanagementportfolio python=3.11
conda activate riskmanagementportfolio

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Ejecucion del Pipeline

```bash
# Pipeline completo (~23 segundos)
python -m pipeline.run_pipeline --all

# Etapas especificas
python -m pipeline.run_pipeline --stages 1,2,3
python -m pipeline.run_pipeline --stages 4,5

# Reentrenamiento (etapas 2-5)
python -m pipeline.run_pipeline --retrain

# Ver estado
python -m pipeline.run_pipeline --status
```

### Ejecucion de la Aplicacion Web

```bash
# Desde la raiz del proyecto
streamlit run streamlit_app/app.py

# Con puerto especifico
streamlit run streamlit_app/app.py --server.port 8501
```

La aplicacion estara disponible en `http://localhost:8501`

### Despliegue en Streamlit Cloud

| Campo | Valor |
|-------|-------|
| **Repository** | `fantastic1121/stocks_portfolio_selector` |
| **Branch** | `main` |
| **Main file path** | `streamlit_app/app.py` |

---

## 11. Autor

**Juan Carlos Ruiz Arteaga**

- GitHub: [@fantastic1121](https://github.com/fantastic1121)
- Repositorio: [stocks_portfolio_selector](https://github.com/fantastic1121/stocks_portfolio_selector)
- Proyecto desarrollado para el curso de Gestion de Riesgo 2025

---

## 12. Disclaimer

> **ADVERTENCIA LEGAL**: Este proyecto es unicamente con fines educativos y de investigacion.
>
> - No constituye asesoria de inversion ni recomendacion de compra o venta de valores
> - Los rendimientos pasados no garantizan resultados futuros
> - Toda inversion conlleva riesgo de perdida de capital
> - Consulte a un asesor financiero certificado antes de invertir
>
> **Marco Regulatorio Colombia**: Las inversiones en valores estan reguladas por la Superintendencia Financiera de Colombia bajo el Decreto 2555 de 2010 y la Ley 964 de 2005.

---

## Licencia

Este proyecto esta bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para mas detalles.

---

<p align="center">
  <i>Portfolio Construction via Clustering - Risk Management 2025</i>
</p>
