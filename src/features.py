# ================================================
# Features Module
# ================================================
"""
Cálculo de métricas financieras de riesgo y retorno.
"""

import numpy as np
import pandas as pd
from typing import Tuple


# Constantes
TRADING_DAYS = 252
RISK_FREE_RATE = 0.02  # 2% anual


def calculate_returns(prices: pd.Series, period: str = 'daily') -> pd.Series:
    """
    Calcular retornos simples.
    
    Args:
        prices: Serie de precios
        period: 'daily' o 'monthly'
        
    Returns:
        Serie de retornos
    """
    if period == 'daily':
        return prices.pct_change().dropna()
    elif period == 'monthly':
        return prices.resample('ME').last().pct_change().dropna()
    else:
        raise ValueError(f"Período no válido: {period}")


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Calcular volatilidad (desviación estándar).
    
    Args:
        returns: Serie de retornos
        annualize: Si True, anualiza la volatilidad
        
    Returns:
        Volatilidad
    """
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(TRADING_DAYS)
    return vol


def calculate_sharpe(returns: pd.Series, 
                     risk_free_rate: float = RISK_FREE_RATE) -> float:
    """
    Calcular Sharpe Ratio anualizado.
    
    Args:
        returns: Serie de retornos diarios
        risk_free_rate: Tasa libre de riesgo anual
        
    Returns:
        Sharpe Ratio
    """
    excess_return = returns.mean() * TRADING_DAYS - risk_free_rate
    volatility = returns.std() * np.sqrt(TRADING_DAYS)
    return excess_return / volatility if volatility > 0 else 0


def calculate_sortino(returns: pd.Series, 
                      risk_free_rate: float = RISK_FREE_RATE) -> float:
    """
    Calcular Sortino Ratio (solo penaliza downside volatility).
    
    Args:
        returns: Serie de retornos diarios
        risk_free_rate: Tasa libre de riesgo anual
        
    Returns:
        Sortino Ratio
    """
    excess_return = returns.mean() * TRADING_DAYS - risk_free_rate
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS)
    return excess_return / downside_std if downside_std > 0 else 0


def calculate_calmar(prices: pd.Series) -> float:
    """
    Calcular Calmar Ratio (retorno anual / max drawdown).
    
    Args:
        prices: Serie de precios
        
    Returns:
        Calmar Ratio
    """
    returns = prices.pct_change().dropna()
    annual_return = returns.mean() * TRADING_DAYS
    max_dd = calculate_max_drawdown(prices)
    return annual_return / abs(max_dd) if max_dd != 0 else 0


def calculate_beta(returns: pd.Series, 
                   benchmark_returns: pd.Series) -> float:
    """
    Calcular Beta respecto al benchmark.
    
    Args:
        returns: Serie de retornos del activo
        benchmark_returns: Serie de retornos del benchmark
        
    Returns:
        Beta
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    variance = aligned.iloc[:, 1].var()
    return covariance / variance if variance > 0 else 0


def calculate_var_cvar(returns: pd.Series, 
                       confidence: float = 0.05) -> Tuple[float, float]:
    """
    Calcular VaR y CVaR histórico.
    
    Args:
        returns: Serie de retornos
        confidence: Nivel de confianza (default: 5%)
        
    Returns:
        Tupla (VaR, CVaR)
    """
    var = returns.quantile(confidence)
    cvar = returns[returns <= var].mean()
    return var, cvar


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calcular Maximum Drawdown.
    
    Args:
        prices: Serie de precios
        
    Returns:
        Maximum Drawdown (valor negativo)
    """
    rolling_max = prices.expanding().max()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown.min()


def calculate_all_features(prices: pd.Series,
                           benchmark_returns: pd.Series,
                           risk_free_rate: float = RISK_FREE_RATE) -> dict:
    """
    Calcular todas las métricas para un activo.
    
    Args:
        prices: Serie de precios del activo
        benchmark_returns: Serie de retornos del benchmark
        risk_free_rate: Tasa libre de riesgo
        
    Returns:
        Diccionario con todas las métricas
    """
    returns = calculate_returns(prices, 'daily')
    var, cvar = calculate_var_cvar(returns)
    
    return {
        'annual_return': returns.mean() * TRADING_DAYS,
        'annual_volatility': calculate_volatility(returns),
        'sharpe_ratio': calculate_sharpe(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino(returns, risk_free_rate),
        'calmar_ratio': calculate_calmar(prices),
        'beta': calculate_beta(returns, benchmark_returns),
        'correlation_spy': returns.corr(benchmark_returns),
        'var_95': var,
        'cvar_95': cvar,
        'max_drawdown': calculate_max_drawdown(prices)
    }
