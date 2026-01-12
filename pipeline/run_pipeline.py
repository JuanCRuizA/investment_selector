#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================
# Pipeline Principal: Orquestador CLI
# ================================================
"""
Orquestador principal del pipeline de construcción de portafolios.

Uso:
    python -m pipeline.run_pipeline --all              # Ejecutar todo el pipeline
    python -m pipeline.run_pipeline --stages 1,2,3    # Ejecutar pasos específicos
    python -m pipeline.run_pipeline --stages 4,5      # Solo portafolios y reportes
    python -m pipeline.run_pipeline --status          # Ver estado del pipeline
    python -m pipeline.run_pipeline --retrain         # Ejecutar reentrenamiento completo

Pasos:
    1. Data Ingestion      - Carga y prepara datos
    2. Feature Engineering - Calcula métricas financieras
    3. Clustering          - Segmenta activos (DBSCAN + K-Means)
    4. Portfolio Selection - Construye portafolios y backtest
    5. Generate Reports    - Genera outputs para web app
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import time

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_config, setup_logging, print_step_header,
    print_success, print_info, print_error, print_warning,
    get_path, ensure_directories
)


# ================================================
# DEFINICIÓN DE ETAPAS
# ================================================

PIPELINE_STAGES = {
    1: {
        'name': 'Data Ingestion',
        'description': 'Carga datos desde SQLite, filtra tickers, split train/test',
        'module': 'pipeline.01_data_ingestion',
        'function': 'run_data_ingestion',
        'outputs': ['data/prices_train.csv', 'data/prices_test.csv']
    },
    2: {
        'name': 'Feature Engineering',
        'description': 'Calcula 21 métricas financieras',
        'module': 'pipeline.02_feature_engineering',
        'function': 'run_feature_engineering_step',
        'outputs': ['data/features_matrix.csv']
    },
    3: {
        'name': 'Clustering',
        'description': 'Segmentación con DBSCAN + K-Means',
        'module': 'pipeline.03_clustering',
        'function': 'run_clustering_step',
        'outputs': ['data/segmentacion_final/activos_segmentados_kmeans.csv']
    },
    4: {
        'name': 'Portfolio Selection',
        'description': 'Construcción de portafolios y backtest',
        'module': 'pipeline.04_portfolio_selection',
        'function': 'run_portfolio_selection_step',
        'outputs': ['reports/portafolio_conservador.csv']
    },
    5: {
        'name': 'Generate Reports',
        'description': 'Genera archivos para aplicación web',
        'module': 'pipeline.05_generate_reports',
        'function': 'run_generate_reports',
        'outputs': ['outputs/api/portfolios.csv']
    }
}


def run_stage(stage_num: int, config: dict = None, logger=None) -> bool:
    """
    Ejecutar una etapa específica del pipeline.
    
    Args:
        stage_num: Número de la etapa (1-5)
        config: Configuración del pipeline
        logger: Logger para registro
        
    Returns:
        True si la etapa se ejecutó correctamente
    """
    if stage_num not in PIPELINE_STAGES:
        print_error(f"Etapa {stage_num} no válida. Use 1-5.")
        return False
    
    stage = PIPELINE_STAGES[stage_num]
    
    print_info(f"\n{'='*60}")
    print_info(f"🚀 EJECUTANDO ETAPA {stage_num}: {stage['name']}")
    print_info(f"   {stage['description']}")
    print_info(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        # Importar módulo dinámicamente
        import importlib
        module = importlib.import_module(stage['module'])
        func = getattr(module, stage['function'])
        
        # Ejecutar función
        result = func(config)
        
        elapsed = time.time() - start_time
        print_success(f"\n✅ ETAPA {stage_num} COMPLETADA en {elapsed:.2f}s")
        
        if logger:
            logger.info(f"Stage {stage_num} ({stage['name']}) completed in {elapsed:.2f}s")
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print_error(f"\n❌ ERROR EN ETAPA {stage_num}: {str(e)}")
        
        if logger:
            logger.error(f"Stage {stage_num} failed: {str(e)}")
        
        import traceback
        traceback.print_exc()
        
        return False


def run_pipeline(stages: list = None, config: dict = None) -> dict:
    """
    Ejecutar el pipeline completo o etapas específicas.
    
    Args:
        stages: Lista de etapas a ejecutar (None = todas)
        config: Configuración del pipeline
        
    Returns:
        Diccionario con resultados de cada etapa
    """
    # Setup logging
    logger = setup_logging('pipeline_main')
    
    # Cargar configuración
    if config is None:
        config = load_config('settings')
    
    # Si no se especifican etapas, ejecutar todas
    if stages is None:
        stages = [1, 2, 3, 4, 5]
    
    # Banner inicial
    print("\n" + "="*70)
    print("   🏦 PIPELINE DE CONSTRUCCIÓN DE PORTAFOLIOS")
    print("   Portfolio Construction via Clustering")
    print("="*70)
    print(f"\n📅 Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Etapas a ejecutar: {stages}")
    
    logger.info(f"Pipeline started with stages: {stages}")
    
    # Asegurar directorios
    ensure_directories(config)
    
    # Ejecutar etapas
    results = {}
    total_start = time.time()
    
    for stage_num in sorted(stages):
        success = run_stage(stage_num, config, logger)
        results[stage_num] = {
            'name': PIPELINE_STAGES[stage_num]['name'],
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if not success:
            print_warning(f"\n⚠️ Pipeline interrumpido en etapa {stage_num}")
            logger.warning(f"Pipeline stopped at stage {stage_num}")
            break
    
    total_elapsed = time.time() - total_start
    
    # Resumen final
    print("\n" + "="*70)
    print("   📊 RESUMEN DE EJECUCIÓN")
    print("="*70)
    
    for stage_num, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"   {status} Etapa {stage_num}: {result['name']}")
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"\n   ⏱️  Tiempo total: {total_elapsed:.2f}s")
    print(f"   📈 Completadas: {successful}/{total} etapas")
    
    if successful == total:
        print("\n🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info(f"Pipeline completed successfully in {total_elapsed:.2f}s")
    else:
        print("\n⚠️ PIPELINE COMPLETADO CON ERRORES")
        logger.warning(f"Pipeline completed with errors in {total_elapsed:.2f}s")
    
    print("="*70 + "\n")
    
    return results


def check_pipeline_status() -> dict:
    """
    Verificar el estado de los archivos del pipeline.
    
    Returns:
        Diccionario con estado de cada etapa
    """
    print("\n" + "="*70)
    print("   📋 ESTADO DEL PIPELINE")
    print("="*70)
    
    status = {}
    
    for stage_num, stage in PIPELINE_STAGES.items():
        outputs = stage['outputs']
        exists = all(get_path(f).exists() for f in outputs)
        
        if exists:
            # Obtener fecha de modificación del primer output
            first_output = get_path(outputs[0])
            mod_time = datetime.fromtimestamp(first_output.stat().st_mtime)
            age = datetime.now() - mod_time
            
            status_str = f"✅ Completada ({mod_time.strftime('%Y-%m-%d %H:%M')})"
            
            # Advertir si es muy antiguo
            if age.days > 180:  # 6 meses
                status_str += " ⚠️ >6 meses"
        else:
            status_str = "❌ No ejecutada"
        
        status[stage_num] = {
            'name': stage['name'],
            'exists': exists,
            'status': status_str
        }
        
        print(f"\n   Etapa {stage_num}: {stage['name']}")
        print(f"   Estado: {status_str}")
    
    print("\n" + "="*70 + "\n")
    
    return status


def main():
    """
    Punto de entrada principal del CLI.
    """
    parser = argparse.ArgumentParser(
        description='Pipeline de Construcción de Portafolios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python -m pipeline.run_pipeline --all           # Ejecutar todo
  python -m pipeline.run_pipeline --stages 1,2    # Solo data + features
  python -m pipeline.run_pipeline --stages 4,5    # Solo portafolios + reportes
  python -m pipeline.run_pipeline --status        # Ver estado
  python -m pipeline.run_pipeline --retrain       # Reentrenamiento completo
        """
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Ejecutar todas las etapas del pipeline'
    )
    
    parser.add_argument(
        '--stages', '-s',
        type=str,
        help='Etapas específicas a ejecutar (ej: 1,2,3 o 4-5)'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Mostrar estado actual del pipeline'
    )
    
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Ejecutar reentrenamiento completo (etapas 2-5)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Ruta a archivo de configuración alternativo'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = None
    if args.config:
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # Procesar comandos
    if args.status:
        check_pipeline_status()
        return 0
    
    if args.all:
        results = run_pipeline(None, config)
        return 0 if all(r['success'] for r in results.values()) else 1
    
    if args.retrain:
        # Reentrenamiento: etapas 2-5 (asume que los datos ya están)
        print_info("🔄 Modo RETRAIN: Ejecutando etapas 2-5")
        results = run_pipeline([2, 3, 4, 5], config)
        return 0 if all(r['success'] for r in results.values()) else 1
    
    if args.stages:
        # Parsear etapas
        stages = []
        for part in args.stages.split(','):
            if '-' in part:
                # Rango (ej: 2-4)
                start, end = map(int, part.split('-'))
                stages.extend(range(start, end + 1))
            else:
                stages.append(int(part))
        
        results = run_pipeline(sorted(set(stages)), config)
        return 0 if all(r['success'] for r in results.values()) else 1
    
    # Si no se especifica nada, mostrar ayuda
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
