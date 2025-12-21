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
