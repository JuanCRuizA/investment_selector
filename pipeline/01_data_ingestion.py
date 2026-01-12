#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================
# Pipeline Step 01: Data Ingestion
# ================================================
"""
Paso 1 del pipeline: Carga de datos desde SQLite, filtrado de tickers
válidos, imputación de valores y split train/test.

Entradas:
    - data/raw/trading_data.db (SQLite)
    
Salidas:
    - data/prices_train.csv
    - data/prices_test.csv
    - reports/valid_tickers.csv
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.utils import (
    load_config, get_path, print_step_header, print_success, 
    print_info, print_error, ensure_directories
)
from src.data_loader import (
    connect_database, load_prices, get_valid_tickers, pivot_prices,
    impute_adj_close, split_train_test, fill_missing_prices,
    validate_benchmark, save_prices_to_csv, get_data_summary
)


def run_data_ingestion(config: dict = None) -> tuple:
    """
    Ejecutar proceso completo de ingesta de datos.
    
    Args:
        config: Configuración (si None, carga desde archivo)
        
    Returns:
        Tupla (df_train, df_test, valid_tickers)
    """
    print_step_header("DATA INGESTION", 1)
    
    # Cargar configuración
    if config is None:
        config = load_config('settings')
    
    # Asegurar directorios
    ensure_directories()
    
    # Extraer configuración
    data_config = config.get('data', {})
    params = config.get('data_params', {})
    
    db_path = get_path(data_config.get('database_path', 'data/raw/trading_data.db'))
    table_name = data_config.get('table_name', 'prices')
    ticker_col = data_config.get('ticker_column', 'ticker')
    date_col = data_config.get('date_column', 'date')
    min_observations = params.get('min_observations', 1260)
    train_end = params.get('train_end_date', '2023-12-31')
    benchmark = params.get('benchmark_ticker', 'SPY')
    fillna_method = params.get('fillna_method', 'ffill')
    
    # Rutas de salida
    output_dir = get_path('data', create_if_missing=True)
    reports_dir = get_path('reports', create_if_missing=True)
    
    # ================================================
    # 1. CONECTAR A BASE DE DATOS
    # ================================================
    print_info(f"Conectando a: {db_path}")
    
    if not db_path.exists():
        print_error(f"Base de datos no encontrada: {db_path}")
        print_info("Por favor, coloque el archivo trading_data.db en data/raw/")
        raise FileNotFoundError(f"Base de datos no encontrada: {db_path}")
    
    conn = connect_database(str(db_path))
    print_success("Conexión establecida")
    
    # ================================================
    # 2. CARGAR DATOS
    # ================================================
    print_info(f"Cargando tabla: {table_name}")
    df_raw = load_prices(conn, table_name)
    print_success(f"Cargados {len(df_raw):,} registros")
    
    # ================================================
    # 3. IMPUTAR ADJ_CLOSE
    # ================================================
    print_info("Imputando valores nulos de adj_close...")
    df_raw = impute_adj_close(df_raw)
    
    # ================================================
    # 4. FILTRAR TICKERS VÁLIDOS
    # ================================================
    print_info(f"Filtrando tickers con ≥{min_observations} observaciones (~5 años)...")
    valid_tickers = get_valid_tickers(df_raw, min_observations, date_col=date_col, ticker_col=ticker_col)
    print_success(f"Tickers válidos: {len(valid_tickers)}")
    
    # Guardar lista de tickers válidos
    df_tickers = pd.DataFrame({'ticker': valid_tickers})
    df_tickers.to_csv(reports_dir / 'valid_tickers.csv', index=False)
    print_info(f"Lista de tickers guardada en: reports/valid_tickers.csv")
    
    # ================================================
    # 5. FILTRAR Y PIVOTAR
    # ================================================
    print_info("Pivotando datos a formato matriz...")
    df_filtered = df_raw[df_raw[ticker_col].isin(valid_tickers)]
    df_prices = pivot_prices(df_filtered, date_col=date_col, ticker_col=ticker_col)
    print_success(f"Matriz de precios: {df_prices.shape[0]} fechas × {df_prices.shape[1]} tickers")
    
    # ================================================
    # 6. RELLENAR VALORES FALTANTES
    # ================================================
    print_info(f"Rellenando valores faltantes (método: {fillna_method})...")
    missing_before = df_prices.isnull().sum().sum()
    df_prices = fill_missing_prices(df_prices, method=fillna_method)
    missing_after = df_prices.isnull().sum().sum()
    print_success(f"Valores imputados: {missing_before - missing_after:,}")
    
    # ================================================
    # 7. VALIDAR BENCHMARK
    # ================================================
    if not validate_benchmark(df_prices, benchmark):
        print_error(f"Benchmark {benchmark} no válido")
    else:
        print_success(f"Benchmark {benchmark} validado")
    
    # ================================================
    # 8. SPLIT TRAIN/TEST
    # ================================================
    print_info(f"Dividiendo datos (train hasta {train_end})...")
    df_train, df_test = split_train_test(df_prices, train_end)
    
    print_success(f"Train: {df_train.index.min().strftime('%Y-%m-%d')} a {df_train.index.max().strftime('%Y-%m-%d')} ({len(df_train)} días)")
    print_success(f"Test:  {df_test.index.min().strftime('%Y-%m-%d')} a {df_test.index.max().strftime('%Y-%m-%d')} ({len(df_test)} días)")
    
    # ================================================
    # 9. GUARDAR ARCHIVOS
    # ================================================
    train_file = output_dir / data_config.get('prices_train_file', 'prices_train.csv')
    test_file = output_dir / data_config.get('prices_test_file', 'prices_test.csv')
    
    save_prices_to_csv(df_train, train_file)
    save_prices_to_csv(df_test, test_file)
    
    print_success(f"Datos de entrenamiento: {train_file}")
    print_success(f"Datos de prueba: {test_file}")
    
    # ================================================
    # 10. RESUMEN
    # ================================================
    summary = get_data_summary(df_prices)
    print_info(f"\n📊 Resumen de datos:")
    print_info(f"   Total tickers: {summary['n_tickers']}")
    print_info(f"   Total fechas: {summary['n_dates']}")
    print_info(f"   Rango: {summary['date_range']['start']} a {summary['date_range']['end']}")
    
    conn.close()
    
    print_success("\n✅ PASO 1 COMPLETADO: Data Ingestion")
    
    return df_train, df_test, valid_tickers


if __name__ == '__main__':
    # Ejecutar como script independiente
    try:
        df_train, df_test, valid_tickers = run_data_ingestion()
        print(f"\n🎉 Ingesta completada exitosamente")
        print(f"   - {len(df_train.columns)} tickers en train")
        print(f"   - {len(df_test.columns)} tickers en test")
    except Exception as e:
        print_error(f"Error en ingesta de datos: {e}")
        sys.exit(1)
