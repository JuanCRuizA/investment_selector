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


# ================================================
# FUNCIONES NUEVAS PARA PIPELINE PRODUCTIVO
# ================================================

def normalize_min_max(series: pd.Series) -> pd.Series:
    """
    Normalizar serie usando min-max scaling (0-1).
    
    Args:
        series: Serie a normalizar
        
    Returns:
        Serie normalizada
    """
    min_val = series.min()
    max_val = series.max()
    
    if max_val - min_val == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    
    return (series - min_val) / (max_val - min_val)


def calculate_momentum_score(df_segmented: pd.DataFrame,
                             df_prices: pd.DataFrame = None,
                             weights: Dict[str, float] = None,
                             momentum_days: int = 126) -> pd.DataFrame:
    """
    Calcular Score de Momentum para selección de activos.
    
    Fórmula: Score = w1×Return + w2×Momentum_6m + w3×Sharpe + w4×Beta
    
    En mercados alcistas, beta alto es POSITIVO (amplifica ganancias).
    
    Args:
        df_segmented: DataFrame con activos segmentados y métricas
        df_prices: DataFrame de precios (para recalcular momentum si es necesario)
        weights: Diccionario con pesos {'return_annualized': 0.35, ...}
        momentum_days: Días para cálculo de momentum
        
    Returns:
        DataFrame con columna 'score_compuesto' agregada
    """
    df = df_segmented.copy()
    
    # Pesos por defecto
    if weights is None:
        weights = {
            'return_annualized': 0.35,
            'momentum_6m': 0.30,
            'sharpe_ratio': 0.15,
            'beta': 0.20
        }
    
    # Recalcular momentum si no existe y tenemos precios
    if 'momentum_6m' not in df.columns and df_prices is not None:
        momentum_dict = {}
        for ticker in df.index:
            if ticker in df_prices.columns:
                prices = df_prices[ticker].dropna()
                if len(prices) >= momentum_days:
                    momentum_dict[ticker] = (prices.iloc[-1] / prices.iloc[-momentum_days]) - 1
                else:
                    momentum_dict[ticker] = (prices.iloc[-1] / prices.iloc[0]) - 1
            else:
                momentum_dict[ticker] = 0
        df['momentum_6m'] = pd.Series(momentum_dict)
    
    # Normalizar cada métrica (min-max)
    for metric in weights.keys():
        if metric in df.columns:
            df[f'{metric}_norm'] = normalize_min_max(df[metric])
        else:
            df[f'{metric}_norm'] = 0.5  # Valor neutro si falta la métrica
    
    # Calcular score compuesto
    df['score_compuesto'] = sum(
        weights[metric] * df[f'{metric}_norm']
        for metric in weights.keys()
    )
    
    return df


def select_portfolio_by_profile(df_segmented: pd.DataFrame,
                                 profile_config: Dict,
                                 outlier_min_return: float = 0.0,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Seleccionar activos para un perfil de inversión específico.
    
    Args:
        df_segmented: DataFrame con activos segmentados y scores
        profile_config: Configuración del perfil (nombre, distribución)
        outlier_min_return: Retorno mínimo para incluir outliers
        seed: Semilla para reproducibilidad
        
    Returns:
        DataFrame con activos seleccionados y pesos
    """
    np.random.seed(seed)
    
    distribution = profile_config.get('distribution', {})
    
    selected_assets = []
    
    for cluster_id, n_assets in distribution.items():
        cluster_id = int(cluster_id)  # Asegurar tipo int
        
        # Filtrar activos del cluster
        df_cluster = df_segmented[df_segmented['segmento'] == cluster_id].copy()
        
        # FILTRO ESPECIAL: Para outliers (cluster -1), solo incluir rendimiento positivo
        if cluster_id == -1:
            df_cluster = df_cluster[df_cluster['return_annualized'] > outlier_min_return]
            if len(df_cluster) == 0:
                print(f"   ⚠️ No hay outliers con rendimiento > {outlier_min_return:.1%}")
                continue
        
        if len(df_cluster) == 0:
            print(f"   ⚠️ No hay activos en cluster {cluster_id}")
            continue
        
        # Ordenar por score y seleccionar los mejores
        df_cluster = df_cluster.sort_values('score_compuesto', ascending=False)
        
        n_to_select = min(n_assets, len(df_cluster))
        selected = df_cluster.head(n_to_select)
        
        selected_assets.append(selected)
    
    # Combinar todos los activos
    if not selected_assets:
        return pd.DataFrame()
    
    df_portfolio = pd.concat(selected_assets, ignore_index=False)
    
    # Agregar peso equiponderado
    df_portfolio['peso'] = 1 / len(df_portfolio)
    
    return df_portfolio.reset_index()


def build_all_portfolios(df_segmented: pd.DataFrame,
                          profiles_config: Dict,
                          outlier_min_return: float = 0.0) -> Dict[str, pd.DataFrame]:
    """
    Construir todos los portafolios para todos los perfiles.
    
    Args:
        df_segmented: DataFrame con activos segmentados y scores
        profiles_config: Configuración de todos los perfiles
        outlier_min_return: Retorno mínimo para outliers
        
    Returns:
        Diccionario {nombre_perfil: df_portfolio}
    """
    portfolios = {}
    
    profiles = profiles_config.get('profiles', {})
    
    for profile_name, profile_config in profiles.items():
        print(f"   📊 Construyendo portafolio: {profile_name.capitalize()}")
        
        df_portfolio = select_portfolio_by_profile(
            df_segmented=df_segmented,
            profile_config=profile_config,
            outlier_min_return=outlier_min_return
        )
        
        if len(df_portfolio) > 0:
            portfolios[profile_name] = df_portfolio
            print(f"      ✅ {len(df_portfolio)} activos seleccionados")
        else:
            print(f"      ⚠️ No se pudo construir portafolio")
    
    return portfolios


def run_portfolio_selection(df_segmented: pd.DataFrame,
                            df_prices: pd.DataFrame,
                            config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Ejecutar proceso completo de selección de portafolios.
    
    Args:
        df_segmented: DataFrame con activos segmentados
        df_prices: DataFrame de precios (para momentum)
        config: Configuración del pipeline
        
    Returns:
        Diccionario con portafolios por perfil
    """
    from .utils import print_step_header, print_success, print_info, load_config
    
    print_step_header("PORTFOLIO SELECTION", 4)
    
    # Cargar configuración de perfiles
    profiles_config = load_config('profiles')
    
    # Configuración de momentum score
    momentum_config = config.get('momentum_score', {})
    weights = momentum_config.get('weights', None)
    momentum_days = momentum_config.get('momentum_days', 126)
    outlier_min_return = momentum_config.get('outlier_min_return', 0.0)
    
    print_info("Calculando Score de Momentum para todos los activos...")
    print_info(f"Pesos: {weights}")
    
    # Calcular scores
    df_with_scores = calculate_momentum_score(
        df_segmented=df_segmented,
        df_prices=df_prices,
        weights=weights,
        momentum_days=momentum_days
    )
    
    print_success(f"Scores calculados para {len(df_with_scores)} activos")
    
    # Top 5 por score
    top5 = df_with_scores.nlargest(5, 'score_compuesto')[
        ['segmento_nombre', 'return_annualized', 'momentum_6m', 'sharpe_ratio', 'beta', 'score_compuesto']
    ]
    print_info("Top 5 activos por score:")
    print(top5.to_string())
    
    # Construir portafolios
    print_info("\nConstruyendo portafolios por perfil...")
    
    portfolios = build_all_portfolios(
        df_segmented=df_with_scores,
        profiles_config=profiles_config,
        outlier_min_return=outlier_min_return
    )
    
    print_success(f"Portafolios construidos: {list(portfolios.keys())}")
    
    return portfolios, df_with_scores
