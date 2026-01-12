# ================================================
# Clustering Module
# ================================================
"""
Algoritmos de clustering para agrupación de activos.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Tuple, List, Dict
import warnings

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN no está instalado. Usar: pip install hdbscan")


def prepare_features(df_features: pd.DataFrame,
                     feature_cols: List[str],
                     exclude_tickers: List[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Preparar y normalizar features para clustering.
    
    Args:
        df_features: DataFrame con features por activo
        feature_cols: Columnas a usar para clustering
        exclude_tickers: Tickers a excluir (ej: benchmark)
        
    Returns:
        Tupla (DataFrame filtrado, array normalizado)
    """
    df = df_features.copy()
    
    if exclude_tickers:
        df = df.drop(exclude_tickers, errors='ignore')
    
    # Eliminar filas con NaN
    df = df[feature_cols].dropna()
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    return df, X_scaled


def find_optimal_k(X: np.ndarray, 
                   k_range: range = range(2, 15)) -> Dict[str, List]:
    """
    Encontrar número óptimo de clusters usando método del codo.
    
    Args:
        X: Array de features normalizadas
        k_range: Rango de valores de K a probar
        
    Returns:
        Diccionario con inertias y silhouette scores
    """
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
    return {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouettes': silhouettes
    }


def apply_kmeans(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Aplicar K-Means clustering.
    
    Args:
        X: Array de features normalizadas
        n_clusters: Número de clusters
        
    Returns:
        Array de etiquetas de cluster
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(X)


def apply_hierarchical(X: np.ndarray, 
                       n_clusters: int,
                       linkage: str = 'ward') -> np.ndarray:
    """
    Aplicar Hierarchical Clustering.
    
    Args:
        X: Array de features normalizadas
        n_clusters: Número de clusters
        linkage: Método de linkage ('ward', 'complete', 'average', 'single')
        
    Returns:
        Array de etiquetas de cluster
    """
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return hierarchical.fit_predict(X)


def apply_hdbscan(X: np.ndarray,
                  min_cluster_size: int = 10,
                  min_samples: int = 5) -> np.ndarray:
    """
    Aplicar HDBSCAN clustering.
    
    Args:
        X: Array de features normalizadas
        min_cluster_size: Tamaño mínimo de cluster
        min_samples: Mínimo de samples para core points
        
    Returns:
        Array de etiquetas de cluster (-1 = outlier)
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("HDBSCAN no está instalado")
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                 min_samples=min_samples)
    return clusterer.fit_predict(X)


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluar calidad del clustering.
    
    Args:
        X: Array de features normalizadas
        labels: Etiquetas de cluster
        
    Returns:
        Diccionario con métricas de evaluación
    """
    # Filtrar outliers para métricas (etiqueta -1 en HDBSCAN)
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    
    n_clusters = len(set(labels_filtered))
    
    if n_clusters < 2:
        return {
            'n_clusters': n_clusters,
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'n_outliers': sum(labels == -1)
        }
    
    return {
        'n_clusters': n_clusters,
        'silhouette': silhouette_score(X_filtered, labels_filtered),
        'davies_bouldin': davies_bouldin_score(X_filtered, labels_filtered),
        'n_outliers': sum(labels == -1)
    }


def compare_methods(X: np.ndarray, n_clusters: int) -> pd.DataFrame:
    """
    Comparar los 3 métodos de clustering.
    
    Args:
        X: Array de features normalizadas
        n_clusters: Número de clusters para K-Means y Hierarchical
        
    Returns:
        DataFrame con comparación de métodos
    """
    results = []
    
    # K-Means
    labels_km = apply_kmeans(X, n_clusters)
    metrics_km = evaluate_clustering(X, labels_km)
    metrics_km['method'] = 'K-Means'
    results.append(metrics_km)
    
    # Hierarchical
    labels_hc = apply_hierarchical(X, n_clusters)
    metrics_hc = evaluate_clustering(X, labels_hc)
    metrics_hc['method'] = 'Hierarchical'
    results.append(metrics_hc)
    
    # HDBSCAN
    if HDBSCAN_AVAILABLE:
        labels_hdb = apply_hdbscan(X)
        metrics_hdb = evaluate_clustering(X, labels_hdb)
        metrics_hdb['method'] = 'HDBSCAN'
        results.append(metrics_hdb)
    
    return pd.DataFrame(results)


# ================================================
# FUNCIONES NUEVAS PARA PIPELINE PRODUCTIVO
# ================================================

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def calculate_optimal_eps(X: np.ndarray, 
                          min_samples: int = 5,
                          percentile: int = 90) -> float:
    """
    Calcular eps óptimo para DBSCAN usando método del codo.
    
    Args:
        X: Array de features normalizadas
        min_samples: Mínimo de samples para DBSCAN
        percentile: Percentil para sugerir eps
        
    Returns:
        Valor de eps sugerido
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    
    # Distancia al k-ésimo vecino más cercano, ordenada
    distances = np.sort(distances[:, min_samples - 1])
    
    # Usar percentil como punto de corte
    eps = np.percentile(distances, percentile)
    
    return eps


def detect_outliers_dbscan(X: np.ndarray,
                           eps: float = None,
                           min_samples: int = 5,
                           eps_percentile: int = 90) -> np.ndarray:
    """
    Detectar outliers usando DBSCAN.
    
    Args:
        X: Array de features normalizadas
        eps: Radio de vecindad (None = calcular automáticamente)
        min_samples: Mínimo de puntos para formar cluster
        eps_percentile: Percentil para cálculo automático de eps
        
    Returns:
        Array booleano (True = outlier)
    """
    if eps is None:
        eps = calculate_optimal_eps(X, min_samples, eps_percentile)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    # Outliers tienen etiqueta -1
    is_outlier = labels == -1
    
    return is_outlier, labels, eps


def run_hybrid_clustering(X: np.ndarray,
                          n_clusters: int = 4,
                          min_samples_dbscan: int = 5,
                          eps_percentile: int = 90,
                          random_state: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    Ejecutar clustering híbrido: DBSCAN para outliers + K-Means para normales.
    
    Args:
        X: Array de features normalizadas
        n_clusters: Número de clusters para K-Means (excluyendo outliers)
        min_samples_dbscan: min_samples para DBSCAN
        eps_percentile: Percentil para cálculo de eps
        random_state: Semilla para reproducibilidad
        
    Returns:
        Tupla (labels finales, diccionario con metadata)
    """
    # 1. Detectar outliers con DBSCAN
    is_outlier, _, eps_used = detect_outliers_dbscan(
        X, 
        min_samples=min_samples_dbscan,
        eps_percentile=eps_percentile
    )
    
    n_outliers = is_outlier.sum()
    
    # 2. Aplicar K-Means solo a puntos normales
    X_normal = X[~is_outlier]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels_normal = kmeans.fit_predict(X_normal)
    
    # 3. Combinar etiquetas: outliers = -1, normales = 0, 1, 2, 3...
    final_labels = np.full(len(X), -1, dtype=int)
    final_labels[~is_outlier] = labels_normal
    
    # 4. Metadata
    metadata = {
        'n_outliers': n_outliers,
        'pct_outliers': n_outliers / len(X) * 100,
        'eps_used': eps_used,
        'n_clusters': n_clusters,
        'cluster_sizes': {
            -1: n_outliers,
            **{i: (labels_normal == i).sum() for i in range(n_clusters)}
        }
    }
    
    return final_labels, metadata


def assign_segment_names(df_features: pd.DataFrame,
                         labels: np.ndarray,
                         segment_names: Dict[int, str] = None) -> pd.DataFrame:
    """
    Asignar nombres descriptivos a los segmentos basados en características.
    
    Args:
        df_features: DataFrame con features de los activos
        labels: Array de etiquetas de cluster
        segment_names: Diccionario opcional de mapeo {cluster_id: nombre}
        
    Returns:
        DataFrame con columnas 'segmento' y 'segmento_nombre' agregadas
    """
    df = df_features.copy()
    df['segmento'] = labels
    
    if segment_names is None:
        # Nombres por defecto basados en el análisis típico
        segment_names = {
            -1: 'Outliers',
            0: 'Conservador',
            1: 'Alto Rendimiento',
            2: 'Moderado',
            3: 'Estable'
        }
    
    df['segmento_nombre'] = df['segmento'].map(segment_names)
    
    # Si hay segmentos sin nombre, usar genérico
    df['segmento_nombre'] = df['segmento_nombre'].fillna(
        df['segmento'].apply(lambda x: f'Cluster_{x}')
    )
    
    return df


def run_pca_reduction(X: np.ndarray, 
                      n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """
    Reducir dimensionalidad con PCA para visualización.
    
    Args:
        X: Array de features normalizadas
        n_components: Número de componentes (default: 2 para visualización)
        
    Returns:
        Tupla (X_reducido, objeto PCA ajustado)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    return X_pca, pca


def get_cluster_summary(df_segmented: pd.DataFrame,
                        feature_cols: List[str]) -> pd.DataFrame:
    """
    Generar resumen estadístico por cluster/segmento.
    
    Args:
        df_segmented: DataFrame con segmentación
        feature_cols: Columnas de features a resumir
        
    Returns:
        DataFrame con estadísticas por segmento
    """
    summary = df_segmented.groupby(['segmento', 'segmento_nombre']).agg({
        **{col: ['mean', 'std', 'min', 'max'] for col in feature_cols if col in df_segmented.columns},
        'segmento': 'count'
    })
    
    # Aplanar columnas multi-nivel
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'segmento_count': 'n_activos'})
    
    return summary.reset_index()


def run_clustering_pipeline(df_features: pd.DataFrame,
                            config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Ejecutar pipeline completo de clustering.
    
    Args:
        df_features: DataFrame con matriz de features
        config: Diccionario de configuración
        
    Returns:
        Tupla (DataFrame segmentado, metadata del clustering)
    """
    from .utils import print_step_header, print_success, print_info, print_warning
    
    print_step_header("CLUSTERING ANALYSIS", 3)
    
    # Extraer configuración
    clustering_config = config.get('clustering', {})
    dbscan_config = clustering_config.get('dbscan', {})
    kmeans_config = clustering_config.get('kmeans', {})
    segment_names = clustering_config.get('segment_names', {})
    features_config = config.get('features', {})
    
    # Features para clustering
    feature_cols = features_config.get('clustering_features', [
        'return_annualized', 'volatility_annual', 'sharpe_ratio', 
        'sortino_ratio', 'max_drawdown', 'var_95', 'cvar_95',
        'beta', 'skewness', 'kurtosis'
    ])
    
    # Verificar que las features existen
    available_features = [f for f in feature_cols if f in df_features.columns]
    if len(available_features) < len(feature_cols):
        missing = set(feature_cols) - set(available_features)
        print_warning(f"Features faltantes para clustering: {missing}")
    
    print_info(f"Features para clustering: {available_features}")
    
    # Preparar datos
    df_clean = df_features[available_features].dropna()
    print_info(f"Activos con datos completos: {len(df_clean)} de {len(df_features)}")
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # Ejecutar clustering híbrido
    n_clusters = kmeans_config.get('n_clusters', 4)
    min_samples = dbscan_config.get('min_samples', 5)
    eps_percentile = dbscan_config.get('eps_percentile', 90)
    random_state = kmeans_config.get('random_state', 42)
    
    print_info(f"Ejecutando DBSCAN (outliers) + K-Means (K={n_clusters})...")
    
    labels, metadata = run_hybrid_clustering(
        X_scaled,
        n_clusters=n_clusters,
        min_samples_dbscan=min_samples,
        eps_percentile=eps_percentile,
        random_state=random_state
    )
    
    print_success(f"Outliers detectados: {metadata['n_outliers']} ({metadata['pct_outliers']:.1f}%)")
    print_success(f"Clusters formados: {n_clusters}")
    
    # Evaluar clustering
    eval_metrics = evaluate_clustering(X_scaled, labels)
    metadata['evaluation'] = eval_metrics
    print_info(f"Silhouette Score: {eval_metrics['silhouette']:.3f}")
    
    # Asignar nombres de segmentos
    # Convertir claves de string a int si vienen del YAML
    segment_names_int = {int(k): v for k, v in segment_names.items()}
    
    # Crear DataFrame resultado solo con los activos usados
    df_result = df_features.loc[df_clean.index].copy()
    df_result = assign_segment_names(df_result, labels, segment_names_int)
    
    # PCA para visualización
    X_pca, pca = run_pca_reduction(X_scaled)
    df_result['pca_1'] = X_pca[:, 0]
    df_result['pca_2'] = X_pca[:, 1]
    metadata['pca_variance_explained'] = pca.explained_variance_ratio_.tolist()
    
    print_success(f"Segmentación completada: {len(df_result)} activos")
    
    return df_result, metadata
