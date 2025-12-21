# ================================================
# Portfolio Module
# ================================================
"""
Construcción y optimización de carteras de inversión.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional


def select_representatives(df_cluster: pd.DataFrame,
                           cluster_col: str,
                           metric_col: str = 'sharpe_ratio',
                           ascending: bool = False) -> pd.DataFrame:
    """
    Seleccionar el mejor representante de cada cluster.
    
    Args:
        df_cluster: DataFrame con asignaciones de cluster y métricas
        cluster_col: Nombre de columna con etiquetas de cluster
        metric_col: Métrica para selección (default: sharpe_ratio)
        ascending: Si True, selecciona el menor valor
        
    Returns:
        DataFrame con representantes seleccionados
    """
    representatives = []
    
    for cluster_id in df_cluster[cluster_col].unique():
        if cluster_id == -1:  # Ignorar outliers de HDBSCAN
            continue
            
        cluster_assets = df_cluster[df_cluster[cluster_col] == cluster_id]
        
        if ascending:
            best_idx = cluster_assets[metric_col].idxmin()
        else:
            best_idx = cluster_assets[metric_col].idxmax()
        
        representatives.append({
            'ticker': best_idx,
            'cluster': cluster_id,
            metric_col: cluster_assets.loc[best_idx, metric_col]
        })
    
    return pd.DataFrame(representatives)


def portfolio_return(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """
    Calcular retorno anualizado del portfolio.
    
    Args:
        weights: Array de pesos
        returns: DataFrame de retornos diarios
        
    Returns:
        Retorno anualizado
    """
    return np.sum(returns.mean() * weights) * 252


def portfolio_volatility(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """
    Calcular volatilidad anualizada del portfolio.
    
    Args:
        weights: Array de pesos
        returns: DataFrame de retornos diarios
        
    Returns:
        Volatilidad anualizada
    """
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))


def portfolio_sharpe(weights: np.ndarray, 
                     returns: pd.DataFrame,
                     risk_free_rate: float = 0.02) -> float:
    """
    Calcular Sharpe Ratio del portfolio.
    
    Args:
        weights: Array de pesos
        returns: DataFrame de retornos diarios
        risk_free_rate: Tasa libre de riesgo
        
    Returns:
        Sharpe Ratio
    """
    ret = portfolio_return(weights, returns)
    vol = portfolio_volatility(weights, returns)
    return (ret - risk_free_rate) / vol if vol > 0 else 0


def negative_sharpe(weights: np.ndarray, 
                    returns: pd.DataFrame,
                    risk_free_rate: float = 0.02) -> float:
    """
    Sharpe negativo para minimización.
    """
    return -portfolio_sharpe(weights, returns, risk_free_rate)


def optimize_portfolio(returns: pd.DataFrame,
                       min_weight: float = 0.05,
                       max_weight: float = 0.20,
                       risk_free_rate: float = 0.02) -> Tuple[np.ndarray, Dict]:
    """
    Optimizar pesos del portfolio maximizando Sharpe Ratio.
    
    Args:
        returns: DataFrame de retornos diarios (columnas = tickers)
        min_weight: Peso mínimo por activo
        max_weight: Peso máximo por activo (anti-concentración)
        risk_free_rate: Tasa libre de riesgo
        
    Returns:
        Tupla (array de pesos óptimos, diccionario con métricas)
    """
    n_assets = returns.shape[1]
    
    # Restricciones
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Suma = 1
    ]
    
    # Bounds (min y max por activo)
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    
    # Pesos iniciales (equiponderados)
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimizar
    result = minimize(
        negative_sharpe,
        initial_weights,
        args=(returns, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    optimal_weights = result.x
    
    # Métricas del portfolio óptimo
    metrics = {
        'return': portfolio_return(optimal_weights, returns),
        'volatility': portfolio_volatility(optimal_weights, returns),
        'sharpe': portfolio_sharpe(optimal_weights, returns, risk_free_rate),
        'success': result.success,
        'message': result.message
    }
    
    return optimal_weights, metrics


def equal_weight_portfolio(tickers: List[str]) -> Dict[str, float]:
    """
    Crear portfolio equiponderado.
    
    Args:
        tickers: Lista de tickers
        
    Returns:
        Diccionario ticker -> peso
    """
    weight = 1 / len(tickers)
    return {ticker: weight for ticker in tickers}


def apply_concentration_rules(weights: Dict[str, float],
                              cluster_assignments: Dict[str, int],
                              max_per_asset: float = 0.20,
                              max_per_cluster: float = 0.40) -> Dict[str, float]:
    """
    Aplicar reglas anti-concentración a los pesos.
    
    Args:
        weights: Diccionario ticker -> peso
        cluster_assignments: Diccionario ticker -> cluster
        max_per_asset: Máximo peso por activo individual
        max_per_cluster: Máximo peso por cluster
        
    Returns:
        Diccionario con pesos ajustados
    """
    adjusted = weights.copy()
    
    # Limitar por activo individual
    for ticker in adjusted:
        if adjusted[ticker] > max_per_asset:
            adjusted[ticker] = max_per_asset
    
    # Limitar por cluster
    cluster_weights = {}
    for ticker, cluster in cluster_assignments.items():
        if ticker in adjusted:
            cluster_weights.setdefault(cluster, 0)
            cluster_weights[cluster] += adjusted[ticker]
    
    for cluster, total_weight in cluster_weights.items():
        if total_weight > max_per_cluster:
            scale_factor = max_per_cluster / total_weight
            for ticker, c in cluster_assignments.items():
                if c == cluster and ticker in adjusted:
                    adjusted[ticker] *= scale_factor
    
    # Renormalizar para que sume 1
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v/total for k, v in adjusted.items()}
    
    return adjusted
