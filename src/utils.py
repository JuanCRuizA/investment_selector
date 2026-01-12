# ================================================
# Utils Module
# ================================================
"""
Funciones utilitarias comunes para el pipeline.
Incluye manejo de configuración, logging y paths.
"""

import os
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def get_project_root() -> Path:
    """
    Obtener la raíz del proyecto.
    
    Returns:
        Path al directorio raíz del proyecto
    """
    # Buscar hacia arriba hasta encontrar el directorio con config/
    current = Path(__file__).resolve().parent
    
    while current != current.parent:
        if (current / 'config').exists():
            return current
        current = current.parent
    
    # Fallback: asumir que estamos en src/
    return Path(__file__).resolve().parent.parent


def load_config(config_name: str = 'settings') -> Dict[str, Any]:
    """
    Cargar archivo de configuración YAML.
    
    Args:
        config_name: Nombre del archivo (sin extensión), ej: 'settings', 'profiles'
        
    Returns:
        Diccionario con la configuración
    """
    project_root = get_project_root()
    config_path = project_root / 'config' / f'{config_name}.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_path(relative_path: str, create_if_missing: bool = False) -> Path:
    """
    Obtener ruta absoluta a partir de ruta relativa al proyecto.
    
    Args:
        relative_path: Ruta relativa desde la raíz del proyecto
        create_if_missing: Si True, crea el directorio si no existe
        
    Returns:
        Path absoluto
    """
    project_root = get_project_root()
    full_path = project_root / relative_path
    
    if create_if_missing and not full_path.exists():
        if full_path.suffix:  # Es un archivo
            full_path.parent.mkdir(parents=True, exist_ok=True)
        else:  # Es un directorio
            full_path.mkdir(parents=True, exist_ok=True)
    
    return full_path


def setup_logging(
    name: str = 'pipeline',
    level: str = 'INFO',
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Configurar logging para el pipeline.
    
    Args:
        name: Nombre del logger
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Ruta al archivo de log (opcional)
        console_output: Si True, también imprime en consola
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler de consola
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Handler de archivo
    if log_file:
        log_path = get_path(log_file, create_if_missing=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_versioned_filename(
    base_name: str,
    extension: str = 'csv',
    format_type: str = 'timestamp'
) -> str:
    """
    Generar nombre de archivo con versionamiento.
    
    Args:
        base_name: Nombre base del archivo
        extension: Extensión del archivo
        format_type: 'timestamp' o 'date'
        
    Returns:
        Nombre de archivo con versión
    """
    if format_type == 'timestamp':
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:  # date
        version = datetime.now().strftime('%Y%m%d')
    
    return f"{base_name}_{version}.{extension}"


def save_artifact_manifest(
    artifacts: Dict[str, str],
    manifest_path: str = 'data/artifacts_manifest.yaml'
) -> None:
    """
    Guardar manifiesto de artefactos generados.
    
    Args:
        artifacts: Diccionario {nombre: ruta_archivo}
        manifest_path: Ruta del archivo manifiesto
    """
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'artifacts': artifacts
    }
    
    full_path = get_path(manifest_path, create_if_missing=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        yaml.dump(manifest, f, default_flow_style=False, allow_unicode=True)


def load_artifact_manifest(
    manifest_path: str = 'data/artifacts_manifest.yaml'
) -> Dict[str, Any]:
    """
    Cargar manifiesto de artefactos.
    
    Args:
        manifest_path: Ruta del archivo manifiesto
        
    Returns:
        Diccionario con información del manifiesto
    """
    full_path = get_path(manifest_path)
    
    if not full_path.exists():
        return {'generated_at': None, 'artifacts': {}}
    
    with open(full_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def print_step_header(step_name: str, step_number: int = None) -> None:
    """
    Imprimir encabezado de paso del pipeline.
    
    Args:
        step_name: Nombre del paso
        step_number: Número del paso (opcional)
    """
    width = 70
    print("=" * width)
    if step_number:
        print(f"📌 PASO {step_number}: {step_name.upper()}")
    else:
        print(f"📌 {step_name.upper()}")
    print("=" * width)


def print_success(message: str) -> None:
    """Imprimir mensaje de éxito."""
    print(f"✅ {message}")


def print_warning(message: str) -> None:
    """Imprimir mensaje de advertencia."""
    print(f"⚠️ {message}")


def print_error(message: str) -> None:
    """Imprimir mensaje de error."""
    print(f"❌ {message}")


def print_info(message: str) -> None:
    """Imprimir mensaje informativo."""
    print(f"ℹ️ {message}")


def validate_required_files(file_list: list) -> bool:
    """
    Validar que existan los archivos requeridos.
    
    Args:
        file_list: Lista de rutas relativas a validar
        
    Returns:
        True si todos existen, False si falta alguno
    """
    missing = []
    
    for file_path in file_list:
        full_path = get_path(file_path)
        if not full_path.exists():
            missing.append(file_path)
    
    if missing:
        print_error(f"Archivos faltantes: {missing}")
        return False
    
    return True


def ensure_directories(config: Dict[str, Any] = None) -> None:
    """
    Crear todos los directorios necesarios para el pipeline.
    
    Args:
        config: Configuración (opcional, usa valores por defecto si no se provee)
    """
    # Directorios base (siempre necesarios)
    directories = [
        'data/raw',
        'data/processed',
        'data/segmentacion_final',
        'reports',
        'reports/figures',
        'outputs/api',
        'logs'
    ]
    
    # Agregar directorios de configuración si se provee
    if config:
        data_config = config.get('data', {})
        if 'reports_dir' in data_config:
            directories.append(data_config['reports_dir'])
        if 'outputs_dir' in data_config:
            directories.append(data_config['outputs_dir'])
    
    for dir_path in directories:
        get_path(dir_path, create_if_missing=True)
    
    print_success("Directorios del pipeline creados/verificados")
