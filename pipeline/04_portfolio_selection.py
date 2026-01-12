#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================
# Pipeline Step 04: Portfolio Selection & Backtest
# ================================================
"""
Paso 4 del pipeline: Selección de portafolios por perfil de inversión
usando Score de Momentum, y backtesting contra SPY.

Entradas:
    - data/segmentacion_final/activos_segmentados_kmeans.csv
    - data/prices_train.csv
    - data/prices_test.csv
    
Salidas:
    - reports/portafolio_[perfil].csv (5 archivos)
    - reports/backtest_metricas_[perfil].csv (5 archivos)
    - reports/backtest_equity_curves_[perfil].csv (5 archivos)
    - reports/backtest_composicion_[perfil].csv (5 archivos)
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
from src.portfolio import (
    calculate_momentum_score, select_portfolio_by_profile,
    build_all_portfolios, run_portfolio_selection
)
from src.backtesting import (
    simular_buy_and_hold, run_backtests_all_profiles,
    generate_backtest_metrics_df, run_backtest_pipeline
)


def run_portfolio_selection_step(config: dict = None) -> tuple:
    """
    Ejecutar proceso completo de selección de portafolios y backtesting.
    
    Args:
        config: Configuración (si None, carga desde archivo)
        
    Returns:
        Tupla (portfolios, backtest_results)
    """
    print_step_header("PORTFOLIO SELECTION & BACKTEST", 4)
    
    # Cargar configuración
    if config is None:
        config = load_config('settings')
    
    profiles_config = load_config('profiles')
    
    # Extraer configuración
    data_config = config.get('data', {})
    momentum_config = config.get('momentum_score', {})
    backtest_config = config.get('backtesting', {})
    data_params = config.get('data_params', {})
    
    # ================================================
    # 1. CARGAR DATOS
    # ================================================
    # 1.1 Segmentación
    seg_file = get_path(f"{data_config.get('segmentation_dir', 'data/segmentacion_final')}/activos_segmentados_kmeans.csv")
    
    if not seg_file.exists():
        print_error(f"Archivo no encontrado: {seg_file}")
        print_info("Ejecute primero el paso 03_clustering.py")
        raise FileNotFoundError(f"Archivo no encontrado: {seg_file}")
    
    print_info(f"Cargando segmentación de: {seg_file}")
    df_segmented = pd.read_csv(seg_file)
    
    if 'ticker' in df_segmented.columns:
        df_segmented = df_segmented.set_index('ticker')
    
    # Excluir benchmark de la selección
    benchmark = data_params.get('benchmark_ticker', 'SPY')
    if benchmark in df_segmented.index:
        df_segmented = df_segmented.drop(benchmark)
    
    print_success(f"Segmentación cargada: {len(df_segmented)} activos")
    
    # 1.2 Precios de entrenamiento
    train_file = get_path(f"data/{data_config.get('prices_train_file', 'prices_train.csv')}")
    print_info(f"Cargando precios train de: {train_file}")
    df_prices_train = load_prices_from_csv(str(train_file))
    
    # 1.3 Precios de prueba
    test_file = get_path(f"data/{data_config.get('prices_test_file', 'prices_test.csv')}")
    print_info(f"Cargando precios test de: {test_file}")
    df_prices_test = load_prices_from_csv(str(test_file))
    
    print_success(f"Precios train: {len(df_prices_train)} días")
    print_success(f"Precios test: {len(df_prices_test)} días")
    
    # ================================================
    # 2. CALCULAR MOMENTUM SCORES
    # ================================================
    print_info("\n📊 Calculando Score de Momentum...")
    
    weights = momentum_config.get('weights', None)
    momentum_days = momentum_config.get('momentum_days', 126)
    outlier_min_return = momentum_config.get('outlier_min_return', 0.0)
    
    print_info(f"   Pesos: {weights}")
    print_info(f"   Momentum días: {momentum_days}")
    
    df_with_scores = calculate_momentum_score(
        df_segmented=df_segmented,
        df_prices=df_prices_train,
        weights=weights,
        momentum_days=momentum_days
    )
    
    print_success(f"Scores calculados para {len(df_with_scores)} activos")
    
    # Top 5 por score
    print_info("\n🏆 Top 5 activos por Score de Momentum:")
    top5 = df_with_scores.nlargest(5, 'score_compuesto')
    for idx, row in top5.iterrows():
        print_info(f"   {idx}: {row['segmento_nombre']} | Score: {row['score_compuesto']:.4f} | Return: {row['return_annualized']:.2%}")
    
    # ================================================
    # 3. CONSTRUIR PORTAFOLIOS
    # ================================================
    print_info("\n📋 Construyendo portafolios por perfil...")
    
    portfolios = build_all_portfolios(
        df_segmented=df_with_scores,
        profiles_config=profiles_config,
        outlier_min_return=outlier_min_return
    )
    
    print_success(f"Portafolios construidos: {list(portfolios.keys())}")
    
    # ================================================
    # 4. GUARDAR PORTAFOLIOS
    # ================================================
    reports_dir = get_path(data_config.get('reports_dir', 'reports'), create_if_missing=True)
    
    for perfil_name, df_portfolio in portfolios.items():
        output_file = reports_dir / f'portafolio_{perfil_name}.csv'
        df_portfolio.to_csv(output_file, index=False)
        print_info(f"   💾 {output_file}")
    
    # ================================================
    # 5. BACKTESTING
    # ================================================
    print_info("\n📈 Ejecutando Backtesting...")
    
    capital_inicial = backtest_config.get('initial_capital', 10000)
    costo_roundtrip = backtest_config.get('total_cost_roundtrip', 0.001)
    risk_free_rate = backtest_config.get('risk_free_rate_backtest', 0.045)
    
    print_info(f"   Capital inicial: ${capital_inicial:,}")
    print_info(f"   Costo round-trip: {costo_roundtrip:.2%}")
    print_info(f"   Período: {df_prices_test.index.min().strftime('%Y-%m-%d')} a {df_prices_test.index.max().strftime('%Y-%m-%d')}")
    
    backtest_results = run_backtests_all_profiles(
        portfolios=portfolios,
        df_prices_test=df_prices_test,
        benchmark_ticker=benchmark,
        capital_inicial=capital_inicial,
        costo_roundtrip=costo_roundtrip,
        risk_free_rate=risk_free_rate
    )
    
    # ================================================
    # 6. GUARDAR RESULTADOS DE BACKTEST
    # ================================================
    print_info("\n💾 Guardando resultados de backtest...")
    
    for perfil_name, result in backtest_results.items():
        # 6.1 Métricas
        df_metrics = generate_backtest_metrics_df(result, perfil_name)
        df_metrics.to_csv(reports_dir / f'backtest_metricas_{perfil_name}.csv', index=False)
        
        # 6.2 Equity curves
        df_equity = pd.DataFrame({
            'Fecha': result['portfolio']['equity_curve'].index,
            f'Portafolio_{perfil_name.capitalize()}': result['portfolio']['equity_curve'].values,
            'SPY_Benchmark': result['benchmark']['equity_curve'].values
        })
        df_equity.to_csv(reports_dir / f'backtest_equity_curves_{perfil_name}.csv', index=False)
        
        # 6.3 Composición (tickers del portafolio)
        df_composicion = pd.DataFrame({
            'ticker': result['portfolio']['tickers'],
            'posicion': list(result['portfolio']['posiciones'].values())
        })
        df_composicion.to_csv(reports_dir / f'backtest_composicion_{perfil_name}.csv', index=False)
        
        # 6.4 Retornos mensuales
        equity = result['portfolio']['equity_curve']
        retornos_mensuales = equity.resample('ME').last().pct_change().dropna()
        df_retornos = pd.DataFrame({
            'Fecha': retornos_mensuales.index,
            'Retorno': retornos_mensuales.values
        })
        df_retornos.to_csv(reports_dir / f'backtest_retornos_mensuales_{perfil_name}.csv', index=False)
        
        print_info(f"   ✅ {perfil_name.capitalize()}: archivos guardados")
    
    # ================================================
    # 7. RESUMEN DE RESULTADOS
    # ================================================
    print_info("\n📊 RESUMEN DE BACKTESTING:")
    print_info("-" * 70)
    print_info(f"{'Perfil':<15} {'Retorno':<12} {'Alpha':<12} {'Sharpe':<10} {'Max DD':<10}")
    print_info("-" * 70)
    
    for perfil_name, result in backtest_results.items():
        port = result['portfolio']
        alpha = result['alpha']
        print_info(f"{perfil_name.capitalize():<15} {port['retorno_total']:>10.2%} {alpha:>+10.2%} {port['sharpe_ratio']:>9.3f} {port['max_drawdown']:>9.2%}")
    
    # Benchmark
    bench = list(backtest_results.values())[0]['benchmark']
    print_info("-" * 70)
    print_info(f"{'SPY (Benchmark)':<15} {bench['retorno_total']:>10.2%} {'---':>10} {bench['sharpe_ratio']:>9.3f} {bench['max_drawdown']:>9.2%}")
    
    print_success("\n✅ PASO 4 COMPLETADO: Portfolio Selection & Backtest")
    
    return portfolios, backtest_results, df_with_scores


if __name__ == '__main__':
    # Ejecutar como script independiente
    try:
        portfolios, backtest_results, df_with_scores = run_portfolio_selection_step()
        print(f"\n🎉 Selección de portafolios y backtest completados")
        print(f"   - {len(portfolios)} perfiles procesados")
        
        # Mejor perfil por alpha
        best_profile = max(backtest_results.items(), key=lambda x: x[1]['alpha'])
        print(f"   - Mejor perfil por Alpha: {best_profile[0].capitalize()} ({best_profile[1]['alpha']:.2%})")
    except Exception as e:
        print_error(f"Error en selección de portafolios: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
