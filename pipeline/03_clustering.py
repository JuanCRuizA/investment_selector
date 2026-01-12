#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================
# Pipeline Step 03: Clustering Analysis
# ================================================
"""
Paso 3 del pipeline: Detección de outliers con DBSCAN y
segmentación con K-Means para agrupar activos.

Entradas:
    - data/features_matrix.csv
    
Salidas:
    - data/segmentacion_final/activos_segmentados_kmeans.csv
    - data/segmentacion_final/resumen_segmentos.csv
    - data/segmentacion_final/tickers_por_segmento.csv
    - data/segmentacion_final/metadata_segmentacion.txt
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
    print_info, print_error
)
from src.clustering import (
    run_clustering_pipeline, get_cluster_summary
)


def run_clustering_step(config: dict = None) -> tuple:
    """
    Ejecutar proceso completo de clustering.
    
    Args:
        config: Configuración (si None, carga desde archivo)
        
    Returns:
        Tupla (df_segmented, metadata)
    """
    print_step_header("CLUSTERING ANALYSIS", 3)
    
    # Cargar configuración
    if config is None:
        config = load_config('settings')
    
    # Extraer configuración
    data_config = config.get('data', {})
    clustering_config = config.get('clustering', {})
    
    # ================================================
    # 1. CARGAR DATOS DE FEATURES
    # ================================================
    features_file = get_path(f"data/{data_config.get('features_file', 'features_matrix.csv')}")
    
    if not features_file.exists():
        print_error(f"Archivo no encontrado: {features_file}")
        print_info("Ejecute primero el paso 02_feature_engineering.py")
        raise FileNotFoundError(f"Archivo no encontrado: {features_file}")
    
    print_info(f"Cargando features de: {features_file}")
    df_features = pd.read_csv(features_file)
    
    # Establecer ticker como índice si existe
    if 'ticker' in df_features.columns:
        df_features = df_features.set_index('ticker')
    
    print_success(f"Features cargadas: {len(df_features)} activos × {len(df_features.columns)} métricas")
    
    # ================================================
    # 2. EJECUTAR CLUSTERING
    # ================================================
    print_info("Ejecutando clustering híbrido (DBSCAN + K-Means)...")
    
    df_segmented, metadata = run_clustering_pipeline(df_features, config)
    
    # ================================================
    # 3. RESUMEN DE SEGMENTOS
    # ================================================
    print_info("\n📊 Distribución de segmentos:")
    
    segment_counts = df_segmented.groupby(['segmento', 'segmento_nombre']).size().reset_index(name='count')
    
    for _, row in segment_counts.iterrows():
        pct = row['count'] / len(df_segmented) * 100
        emoji = '⚠️' if row['segmento'] == -1 else '📊'
        print_info(f"   {emoji} {row['segmento_nombre']} (C{row['segmento']}): {row['count']} activos ({pct:.1f}%)")
    
    # ================================================
    # 4. GUARDAR RESULTADOS
    # ================================================
    seg_dir = get_path(data_config.get('segmentation_dir', 'data/segmentacion_final'), create_if_missing=True)
    
    # 4.1 Activos segmentados (archivo principal)
    df_segmented_save = df_segmented.reset_index()
    df_segmented_save.to_csv(seg_dir / 'activos_segmentados_kmeans.csv', index=False)
    print_success(f"Segmentación guardada: {seg_dir / 'activos_segmentados_kmeans.csv'}")
    
    # 4.2 Resumen de segmentos
    features_for_summary = config.get('features', {}).get('clustering_features', [])
    available_features = [f for f in features_for_summary if f in df_segmented.columns]
    
    # Crear resumen simplificado
    summary = df_segmented.groupby(['segmento', 'segmento_nombre']).agg({
        'return_annualized': ['mean', 'std'],
        'volatility_annual': 'mean',
        'sharpe_ratio': 'mean',
        'beta': 'mean',
        'max_drawdown': 'mean'
    }).round(4)
    
    summary.columns = ['Ret_Mean', 'Ret_Std', 'Vol_Mean', 'Sharpe_Mean', 'Beta_Mean', 'MaxDD_Mean']
    summary = summary.reset_index()
    summary['n_activos'] = df_segmented.groupby('segmento').size().values
    
    summary.to_csv(seg_dir / 'resumen_segmentos.csv', index=False)
    print_success(f"Resumen guardado: {seg_dir / 'resumen_segmentos.csv'}")
    
    # 4.3 Tickers por segmento
    tickers_by_segment = df_segmented.groupby('segmento_nombre')['ticker'].apply(list) if 'ticker' in df_segmented.columns else df_segmented.groupby('segmento_nombre').apply(lambda x: x.index.tolist())
    tickers_df = pd.DataFrame({
        'segmento_nombre': tickers_by_segment.index,
        'tickers': tickers_by_segment.apply(lambda x: ','.join(x) if isinstance(x, list) else str(x)).values
    })
    tickers_df.to_csv(seg_dir / 'tickers_por_segmento.csv', index=False)
    print_success(f"Tickers por segmento: {seg_dir / 'tickers_por_segmento.csv'}")
    
    # 4.4 Metadata
    metadata_text = f"""# Metadata de Segmentación
# Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# ================================================

Total activos procesados: {len(df_segmented)}
Outliers detectados: {metadata['n_outliers']} ({metadata['pct_outliers']:.1f}%)
Eps DBSCAN utilizado: {metadata['eps_used']:.3f}
Número de clusters K-Means: {metadata['n_clusters']}

## Evaluación del Clustering
Silhouette Score: {metadata['evaluation']['silhouette']:.4f}
Davies-Bouldin Score: {metadata['evaluation']['davies_bouldin']:.4f}

## Varianza explicada por PCA
PC1: {metadata['pca_variance_explained'][0]:.2%}
PC2: {metadata['pca_variance_explained'][1]:.2%}
Total (2 componentes): {sum(metadata['pca_variance_explained']):.2%}

## Distribución de Clusters
{json.dumps({k: int(v) for k, v in metadata['cluster_sizes'].items()}, indent=2)}
"""
    
    with open(seg_dir / 'metadata_segmentacion.txt', 'w', encoding='utf-8') as f:
        f.write(metadata_text)
    print_success(f"Metadata: {seg_dir / 'metadata_segmentacion.txt'}")
    
    # ================================================
    # 5. RESUMEN FINAL
    # ================================================
    print_info(f"\n📊 Resumen del clustering:")
    print_info(f"   Método: DBSCAN (outliers) + K-Means (K={metadata['n_clusters']})")
    print_info(f"   Silhouette Score: {metadata['evaluation']['silhouette']:.4f}")
    print_info(f"   Outliers: {metadata['n_outliers']} ({metadata['pct_outliers']:.1f}%)")
    
    print_success("\n✅ PASO 3 COMPLETADO: Clustering Analysis")
    
    return df_segmented, metadata


if __name__ == '__main__':
    # Ejecutar como script independiente
    try:
        df_segmented, metadata = run_clustering_step()
        print(f"\n🎉 Clustering completado exitosamente")
        print(f"   - {len(df_segmented)} activos segmentados")
        print(f"   - {metadata['n_clusters'] + 1} segmentos identificados (incluyendo outliers)")
    except Exception as e:
        print_error(f"Error en clustering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
