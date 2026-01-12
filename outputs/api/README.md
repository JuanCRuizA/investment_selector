# 📁 API Outputs Directory

Este directorio contiene los archivos generados por el pipeline, listos para ser consumidos por la aplicación web.

## Archivos Generados

| Archivo | Descripción | Formato |
|---------|-------------|---------|
| `portfolios.csv` | Composición de todos los portafolios | ticker, perfil, peso, métricas |
| `segments.csv` | Resumen de segmentos de activos | segmento, estadísticas |
| `backtest_summary.csv` | Métricas de backtest consolidadas | perfil, retorno, sharpe, etc. |
| `equity_curves.csv` | Series temporales de equity | fecha, equity, benchmark |
| `metadata.csv` | Información del pipeline | versión, fechas, parámetros |
| `metadata.json` | Mismo contenido en JSON | - |

## Uso en Aplicación Web

```python
import pandas as pd

# Cargar portafolios
portfolios = pd.read_csv('outputs/api/portfolios.csv')

# Filtrar por perfil
conservador = portfolios[portfolios['perfil'] == 'conservador']

# Cargar equity curves
equity = pd.read_csv('outputs/api/equity_curves.csv')
```

## Actualización

Los archivos se regeneran automáticamente al ejecutar:

```bash
python -m pipeline.run_pipeline --all
# o solo los reportes:
python -m pipeline.run_pipeline --stages 5
```

## Formato de Fechas

- Todas las fechas están en formato ISO: `YYYY-MM-DD`
- El campo `generated_at` en metadata usa formato ISO con hora: `YYYY-MM-DDTHH:MM:SS`
