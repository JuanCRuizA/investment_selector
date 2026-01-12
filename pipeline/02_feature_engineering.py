#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================
# Pipeline Step 02: Feature Engineering
# ================================================
"""
Paso 2 del pipeline: Cálculo de las 21 métricas financieras
para cada activo usando los datos de entrenamiento.

Entradas:
    - data/prices_train.csv
    
Salidas:
    - data/features_matrix.csv
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.utils import (
    load_config, get_path, print_step_header, print_success, 
    print_info, print_error
)
from src.data_loader import load_prices_from_csv
from src.features import (
    calculate_features_matrix, run_feature_engineering,
    TRADING_DAYS
)


def run_feature_engineering_step(config: dict = None) -> pd.DataFrame:
    """
    Ejecutar proceso completo de feature engineering.
    
    Args:
        config: Configuración (si None, carga desde archivo)
        
    Returns:
        DataFrame con matriz de features
    """
    print_step_header("FEATURE ENGINEERING", 2)
    
    # Cargar configuración
    if config is None:
        config = load_config('settings')
    
    # Extraer configuración
    data_config = config.get('data', {})
    financial_params = config.get('financial_params', {})
    features_config = config.get('features', {})
    data_params = config.get('data_params', {})
    momentum_config = config.get('momentum_score', {})
    
    # Parámetros
    risk_free_rate = financial_params.get('risk_free_rate', 0.05)
    momentum_days = momentum_config.get('momentum_days', 126)
    benchmark = data_params.get('benchmark_ticker', 'SPY')
    
    # ================================================
    # 1. CARGAR DATOS DE PRECIOS
    # ================================================
    train_file = get_path(f"data/{data_config.get('prices_train_file', 'prices_train.csv')}")
    
    if not train_file.exists():
        print_error(f"Archivo no encontrado: {train_file}")
        print_info("Ejecute primero el paso 01_data_ingestion.py")
        raise FileNotFoundError(f"Archivo no encontrado: {train_file}")
    
    print_info(f"Cargando precios de: {train_file}")
    df_prices = load_prices_from_csv(str(train_file))
    
    print_success(f"Precios cargados: {df_prices.shape[0]} fechas × {df_prices.shape[1]} tickers")
    
    # ================================================
    # 2. CALCULAR FEATURES
    # ================================================
    print_info(f"Calculando 21 métricas financieras...")
    print_info(f"   Risk-free rate: {risk_free_rate:.1%}")
    print_info(f"   Días de trading: {TRADING_DAYS}")
    print_info(f"   Benchmark: {benchmark}")
    print_info(f"   Momentum days: {momentum_days}")
    
    df_features = calculate_features_matrix(
        df_prices=df_prices,
        benchmark_ticker=benchmark,
        risk_free_rate=risk_free_rate,
        momentum_days=momentum_days
    )
    
    print_success(f"Features calculadas para {len(df_features)} activos")
    
    # ================================================
    # 3. ESTADÍSTICAS DE FEATURES
    # ================================================
    print_info("\n📊 Estadísticas de features:")
    
    # Mostrar algunas estadísticas clave
    stats_cols = ['return_annualized', 'volatility_annual', 'sharpe_ratio', 'beta', 'max_drawdown']
    available_stats = [c for c in stats_cols if c in df_features.columns]
    
    for col in available_stats:
        mean_val = df_features[col].mean()
        std_val = df_features[col].std()
        if col in ['return_annualized', 'volatility_annual', 'max_drawdown']:
            print_info(f"   {col}: {mean_val:.2%} (±{std_val:.2%})")
        else:
            print_info(f"   {col}: {mean_val:.3f} (±{std_val:.3f})")
    
    # ================================================
    # 4. GUARDAR RESULTADOS
    # ================================================
    output_file = get_path(f"data/{data_config.get('features_file', 'features_matrix.csv')}")
    
    # Guardar con ticker como columna (no índice)
    df_features_save = df_features.reset_index()
    df_features_save.to_csv(output_file, index=False)
    
    print_success(f"Features guardadas en: {output_file}")
    
    # ================================================
    # 5. RESUMEN
    # ================================================
    print_info(f"\n📊 Resumen de features:")
    print_info(f"   Total activos: {len(df_features)}")
    print_info(f"   Total métricas: {len(df_features.columns)}")
    print_info(f"   Métricas: {list(df_features.columns)}")
    
    # Valores faltantes
    missing = df_features.isnull().sum()
    if missing.sum() > 0:
        print_info(f"\n   ⚠️ Valores faltantes:")
        for col, count in missing[missing > 0].items():
            print_info(f"      {col}: {count}")
    else:
        print_success("   Sin valores faltantes")
    
    print_success("\n✅ PASO 2 COMPLETADO: Feature Engineering")
    
    return df_features


if __name__ == '__main__':
    # Ejecutar como script independiente
    try:
        df_features = run_feature_engineering_step()
        print(f"\n🎉 Feature engineering completado exitosamente")
        print(f"   - {len(df_features)} activos procesados")
        print(f"   - {len(df_features.columns)} métricas calculadas")
    except Exception as e:
        print_error(f"Error en feature engineering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
