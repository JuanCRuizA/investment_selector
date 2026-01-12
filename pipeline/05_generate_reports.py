#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================
# Pipeline Step 05: Generate Reports for Web App
# ================================================
"""
Paso 5 del pipeline: Consolidar resultados y generar archivos
de salida para consumo por la aplicación web.

Entradas:
    - Todos los archivos generados en pasos anteriores
    
Salidas:
    - outputs/api/portfolios.csv
    - outputs/api/segments.csv
    - outputs/api/backtest_summary.csv
    - outputs/api/equity_curves.csv
    - outputs/api/metadata.csv
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from datetime import datetime
from src.utils import (
    load_config, get_path, print_step_header, print_success, 
    print_info, print_error, save_artifact_manifest
)


def run_generate_reports(config: dict = None) -> dict:
    """
    Generar reportes consolidados para la aplicación web.
    
    Args:
        config: Configuración (si None, carga desde archivo)
        
    Returns:
        Diccionario con rutas de archivos generados
    """
    print_step_header("GENERATE REPORTS FOR WEB APP", 5)
    
    # Cargar configuración
    if config is None:
        config = load_config('settings')
    
    profiles_config = load_config('profiles')
    
    # Extraer configuración
    data_config = config.get('data', {})
    perfiles = ['conservador', 'moderado', 'agresivo', 'especulativo', 'normal']
    
    # Rutas
    reports_dir = get_path(data_config.get('reports_dir', 'reports'))
    seg_dir = get_path(data_config.get('segmentation_dir', 'data/segmentacion_final'))
    output_dir = get_path(data_config.get('outputs_dir', 'outputs/api'), create_if_missing=True)
    
    generated_files = {}
    
    # ================================================
    # 1. CONSOLIDAR PORTAFOLIOS
    # ================================================
    print_info("📋 Consolidando portafolios...")
    
    all_portfolios = []
    
    for perfil in perfiles:
        try:
            df_port = pd.read_csv(reports_dir / f'portafolio_{perfil}.csv')
            df_port['perfil'] = perfil
            all_portfolios.append(df_port)
        except FileNotFoundError:
            print_info(f"   ⚠️ No encontrado: portafolio_{perfil}.csv")
    
    if all_portfolios:
        df_portfolios = pd.concat(all_portfolios, ignore_index=True)
        
        # Seleccionar columnas relevantes
        cols_export = ['perfil', 'ticker', 'segmento_nombre', 'return_annualized', 
                       'volatility_annual', 'sharpe_ratio', 'beta', 'momentum_6m', 
                       'score_compuesto', 'peso']
        cols_available = [c for c in cols_export if c in df_portfolios.columns]
        
        df_portfolios[cols_available].to_csv(output_dir / 'portfolios.csv', index=False)
        generated_files['portfolios'] = str(output_dir / 'portfolios.csv')
        print_success(f"   ✅ portfolios.csv ({len(df_portfolios)} registros)")
    
    # ================================================
    # 2. CONSOLIDAR SEGMENTOS
    # ================================================
    print_info("📊 Consolidando información de segmentos...")
    
    try:
        df_segments = pd.read_csv(seg_dir / 'resumen_segmentos.csv')
        
        # Agregar descripción de perfiles
        profile_descriptions = profiles_config.get('profiles', {})
        
        df_segments.to_csv(output_dir / 'segments.csv', index=False)
        generated_files['segments'] = str(output_dir / 'segments.csv')
        print_success(f"   ✅ segments.csv ({len(df_segments)} segmentos)")
    except FileNotFoundError:
        print_info("   ⚠️ No encontrado: resumen_segmentos.csv")
    
    # ================================================
    # 3. CONSOLIDAR MÉTRICAS DE BACKTEST
    # ================================================
    print_info("📈 Consolidando métricas de backtest...")
    
    all_metrics = []
    
    for perfil in perfiles:
        try:
            df_metrics = pd.read_csv(reports_dir / f'backtest_metricas_{perfil}.csv')
            
            # Transponer para formato más útil
            metrics_dict = {'perfil': perfil}
            for _, row in df_metrics.iterrows():
                metrica = row['Metrica']
                metrics_dict[f'{metrica}_portafolio'] = row.iloc[1]  # Columna del portafolio
                metrics_dict[f'{metrica}_benchmark'] = row.iloc[2]  # Columna del benchmark
            
            all_metrics.append(metrics_dict)
        except FileNotFoundError:
            print_info(f"   ⚠️ No encontrado: backtest_metricas_{perfil}.csv")
    
    if all_metrics:
        df_backtest_summary = pd.DataFrame(all_metrics)
        df_backtest_summary.to_csv(output_dir / 'backtest_summary.csv', index=False)
        generated_files['backtest_summary'] = str(output_dir / 'backtest_summary.csv')
        print_success(f"   ✅ backtest_summary.csv ({len(df_backtest_summary)} perfiles)")
    
    # ================================================
    # 4. CONSOLIDAR EQUITY CURVES
    # ================================================
    print_info("📉 Consolidando equity curves...")
    
    equity_dfs = []
    
    for perfil in perfiles:
        try:
            df_equity = pd.read_csv(reports_dir / f'backtest_equity_curves_{perfil}.csv')
            df_equity['perfil'] = perfil
            # Renombrar columnas genéricamente
            df_equity.columns = ['fecha', 'equity_portafolio', 'equity_benchmark', 'perfil']
            equity_dfs.append(df_equity)
        except FileNotFoundError:
            print_info(f"   ⚠️ No encontrado: backtest_equity_curves_{perfil}.csv")
    
    if equity_dfs:
        df_equity_all = pd.concat(equity_dfs, ignore_index=True)
        df_equity_all.to_csv(output_dir / 'equity_curves.csv', index=False)
        generated_files['equity_curves'] = str(output_dir / 'equity_curves.csv')
        print_success(f"   ✅ equity_curves.csv ({len(df_equity_all)} registros)")
    
    # ================================================
    # 5. GENERAR METADATA
    # ================================================
    print_info("📝 Generando metadata...")
    
    # Cargar información de precios para fechas
    try:
        df_train = pd.read_csv(get_path('data/prices_train.csv'), nrows=1, parse_dates=['date'])
        df_test = pd.read_csv(get_path('data/prices_test.csv'), parse_dates=['date'])
        
        # Leer última fila de train para fecha final
        df_train_full = pd.read_csv(get_path('data/prices_train.csv'), parse_dates=['date'])
        train_start = df_train_full['date'].min().strftime('%Y-%m-%d')
        train_end = df_train_full['date'].max().strftime('%Y-%m-%d')
        test_start = df_test['date'].min().strftime('%Y-%m-%d')
        test_end = df_test['date'].max().strftime('%Y-%m-%d')
    except:
        train_start = train_end = test_start = test_end = 'N/A'
    
    # Contar activos
    try:
        df_seg = pd.read_csv(seg_dir / 'activos_segmentados_kmeans.csv')
        n_activos = len(df_seg)
    except:
        n_activos = 0
    
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'project': 'Portfolio Construction via Clustering',
        'version': '1.0',
        'author': 'Juan Carlos Ruiz Arteaga',
        'n_activos': n_activos,
        'n_perfiles': len(perfiles),
        'perfiles': perfiles,
        'train_period': f'{train_start} a {train_end}',
        'test_period': f'{test_start} a {test_end}',
        'benchmark': config.get('data_params', {}).get('benchmark_ticker', 'SPY'),
        'capital_inicial': config.get('backtesting', {}).get('initial_capital', 10000),
        'costo_transaccion': config.get('backtesting', {}).get('total_cost_roundtrip', 0.001),
        'files_generated': list(generated_files.keys())
    }
    
    # Guardar como CSV (más compatible que JSON para algunas apps web)
    df_metadata = pd.DataFrame([metadata])
    df_metadata.to_csv(output_dir / 'metadata.csv', index=False)
    generated_files['metadata'] = str(output_dir / 'metadata.csv')
    print_success(f"   ✅ metadata.csv")
    
    # También guardar como JSON para flexibilidad
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    generated_files['metadata_json'] = str(output_dir / 'metadata.json')
    print_success(f"   ✅ metadata.json")
    
    # ================================================
    # 6. GUARDAR MANIFIESTO DE ARTEFACTOS
    # ================================================
    save_artifact_manifest(generated_files)
    print_success("   ✅ artifacts_manifest.yaml")
    
    # ================================================
    # 7. RESUMEN
    # ================================================
    print_info(f"\n📁 Archivos generados en: {output_dir}")
    for name, path in generated_files.items():
        print_info(f"   • {name}: {Path(path).name}")
    
    print_success("\n✅ PASO 5 COMPLETADO: Generate Reports")
    
    return generated_files


if __name__ == '__main__':
    # Ejecutar como script independiente
    try:
        generated_files = run_generate_reports()
        print(f"\n🎉 Reportes generados exitosamente")
        print(f"   - {len(generated_files)} archivos creados")
        print(f"   - Listos para consumo por aplicación web")
    except Exception as e:
        print_error(f"Error generando reportes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
