# ================================================
# Features Module
# ================================================
"""
Cálculo de métricas financieras de riesgo y retorno.
Incluye todas las 21 métricas usadas en el pipeline de clustering.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from scipy import stats


# Constantes (pueden ser sobrescritas por configuración)
TRADING_DAYS = 252
RISK_FREE_RATE = 0.05  # 5% anual


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


# ================================================
# FUNCIONES NUEVAS PARA PIPELINE PRODUCTIVO
# ================================================

def calculate_alpha(returns: pd.Series,
                    benchmark_returns: pd.Series,
                    risk_free_rate: float = RISK_FREE_RATE) -> float:
    """
    Calcular Alpha de Jensen (exceso de retorno sobre CAPM).
    
    Args:
        returns: Serie de retornos del activo
        benchmark_returns: Serie de retornos del benchmark
        risk_free_rate: Tasa libre de riesgo anual
        
    Returns:
        Alpha anualizado
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    
    asset_returns = aligned.iloc[:, 0]
    market_returns = aligned.iloc[:, 1]
    
    # Calcular beta
    beta = calculate_beta(asset_returns, market_returns)
    
    # Retorno promedio anualizado
    asset_annual_return = asset_returns.mean() * TRADING_DAYS
    market_annual_return = market_returns.mean() * TRADING_DAYS
    
    # Alpha = R_asset - [Rf + Beta * (R_market - Rf)]
    expected_return = risk_free_rate + beta * (market_annual_return - risk_free_rate)
    alpha = asset_annual_return - expected_return
    
    return alpha


def calculate_r_squared(returns: pd.Series,
                        benchmark_returns: pd.Series) -> float:
    """
    Calcular R² (coeficiente de determinación) vs benchmark.
    
    Args:
        returns: Serie de retornos del activo
        benchmark_returns: Serie de retornos del benchmark
        
    Returns:
        R² entre 0 y 1
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    
    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return correlation ** 2


def calculate_skewness(returns: pd.Series) -> float:
    """
    Calcular Skewness (asimetría) de los retornos.
    
    Args:
        returns: Serie de retornos
        
    Returns:
        Skewness (>0 cola derecha, <0 cola izquierda)
    """
    return stats.skew(returns.dropna())


def calculate_kurtosis(returns: pd.Series) -> float:
    """
    Calcular Kurtosis (curtosis) de los retornos.
    
    Args:
        returns: Serie de retornos
        
    Returns:
        Excess Kurtosis (>0 colas pesadas, <0 colas livianas)
    """
    return stats.kurtosis(returns.dropna())


def calculate_positive_return_ratio(returns: pd.Series) -> float:
    """
    Calcular porcentaje de días con retorno positivo.
    
    Args:
        returns: Serie de retornos
        
    Returns:
        Ratio entre 0 y 1
    """
    returns = returns.dropna()
    return (returns > 0).sum() / len(returns) if len(returns) > 0 else 0


def calculate_gain_loss_ratio(returns: pd.Series) -> float:
    """
    Calcular ratio de ganancia promedio / pérdida promedio.
    
    Args:
        returns: Serie de retornos
        
    Returns:
        Ratio (>1 indica ganancias mayores que pérdidas en promedio)
    """
    returns = returns.dropna()
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(gains) == 0 or len(losses) == 0:
        return np.nan
    
    avg_gain = gains.mean()
    avg_loss = abs(losses.mean())
    
    return avg_gain / avg_loss if avg_loss > 0 else np.nan


def calculate_vol_of_vol(returns: pd.Series, window: int = 21) -> float:
    """
    Calcular volatilidad de la volatilidad (rolling).
    
    Args:
        returns: Serie de retornos
        window: Ventana para volatilidad rolling (default: 21 días = 1 mes)
        
    Returns:
        Std de la volatilidad rolling
    """
    rolling_vol = returns.rolling(window).std()
    return rolling_vol.std()


def calculate_downside_deviation(returns: pd.Series,
                                  target: float = 0) -> float:
    """
    Calcular desviación estándar del downside.
    
    Args:
        returns: Serie de retornos
        target: Retorno objetivo (default: 0)
        
    Returns:
        Downside deviation anualizado
    """
    downside = returns[returns < target]
    if len(downside) == 0:
        return 0
    return downside.std() * np.sqrt(TRADING_DAYS)


def calculate_momentum(prices: pd.Series, days: int = 126) -> float:
    """
    Calcular momentum de precio (retorno del período).
    
    Args:
        prices: Serie de precios
        days: Número de días para el cálculo (default: 126 ~ 6 meses)
        
    Returns:
        Momentum como retorno porcentual
    """
    prices = prices.dropna()
    
    if len(prices) < days:
        # Si no hay suficientes datos, usar todo el período disponible
        return (prices.iloc[-1] / prices.iloc[0]) - 1
    
    return (prices.iloc[-1] / prices.iloc[-days]) - 1


def calculate_all_features_extended(prices: pd.Series,
                                     benchmark_prices: pd.Series,
                                     risk_free_rate: float = RISK_FREE_RATE,
                                     momentum_days: int = 126) -> Dict[str, float]:
    """
    Calcular TODAS las 21 métricas para un activo.
    Esta función extiende calculate_all_features con métricas adicionales.
    
    Args:
        prices: Serie de precios del activo
        benchmark_prices: Serie de precios del benchmark
        risk_free_rate: Tasa libre de riesgo
        momentum_days: Días para cálculo de momentum
        
    Returns:
        Diccionario con todas las 21 métricas
    """
    # Calcular retornos
    returns = calculate_returns(prices, 'daily')
    benchmark_returns = calculate_returns(benchmark_prices, 'daily')
    
    # VaR y CVaR
    var, cvar = calculate_var_cvar(returns)
    
    # Retornos totales
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    n_years = len(prices) / TRADING_DAYS
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    return {
        # Retornos
        'return_total': total_return,
        'return_annualized': annual_return,
        'return_mean_daily': returns.mean(),
        
        # Riesgo
        'volatility_annual': calculate_volatility(returns),
        'downside_dev_annual': calculate_downside_deviation(returns),
        'max_drawdown': calculate_max_drawdown(prices),
        'var_95': var,
        'cvar_95': cvar,
        
        # Ratios de eficiencia
        'sharpe_ratio': calculate_sharpe(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino(returns, risk_free_rate),
        'calmar_ratio': calculate_calmar(prices),
        
        # Exposición al mercado
        'beta': calculate_beta(returns, benchmark_returns),
        'alpha_annual': calculate_alpha(returns, benchmark_returns, risk_free_rate),
        'r_squared': calculate_r_squared(returns, benchmark_returns),
        'correlation_spy': returns.corr(benchmark_returns),
        
        # Distribución
        'skewness': calculate_skewness(returns),
        'kurtosis': calculate_kurtosis(returns),
        'positive_return_ratio': calculate_positive_return_ratio(returns),
        'gain_loss_ratio': calculate_gain_loss_ratio(returns),
        'vol_of_vol': calculate_vol_of_vol(returns),
        
        # Momentum
        'momentum_6m': calculate_momentum(prices, momentum_days)
    }


def calculate_features_matrix(df_prices: pd.DataFrame,
                               benchmark_ticker: str = 'SPY',
                               risk_free_rate: float = RISK_FREE_RATE,
                               momentum_days: int = 126) -> pd.DataFrame:
    """
    Calcular matriz de features para todos los tickers.
    
    Args:
        df_prices: DataFrame de precios (índice=fecha, columnas=tickers)
        benchmark_ticker: Ticker del benchmark
        risk_free_rate: Tasa libre de riesgo
        momentum_days: Días para cálculo de momentum
        
    Returns:
        DataFrame con features (índice=ticker, columnas=métricas)
    """
    if benchmark_ticker not in df_prices.columns:
        raise ValueError(f"Benchmark {benchmark_ticker} no encontrado en los datos")
    
    benchmark_prices = df_prices[benchmark_ticker]
    
    features_list = []
    tickers = [col for col in df_prices.columns if col != benchmark_ticker]
    
    for ticker in tickers:
        try:
            prices = df_prices[ticker].dropna()
            
            if len(prices) < 60:  # Mínimo 60 días de datos
                continue
            
            features = calculate_all_features_extended(
                prices=prices,
                benchmark_prices=benchmark_prices,
                risk_free_rate=risk_free_rate,
                momentum_days=momentum_days
            )
            features['ticker'] = ticker
            features_list.append(features)
            
        except Exception as e:
            print(f"   ⚠️ Error calculando features para {ticker}: {e}")
            continue
    
    df_features = pd.DataFrame(features_list)
    df_features = df_features.set_index('ticker')
    
    return df_features


def run_feature_engineering(df_prices: pd.DataFrame, 
                            config: Dict[str, Any]) -> pd.DataFrame:
    """
    Ejecutar proceso completo de feature engineering.
    
    Args:
        df_prices: DataFrame de precios de entrenamiento
        config: Diccionario de configuración
        
    Returns:
        DataFrame con matriz de features
    """
    from .utils import print_step_header, print_success, print_info
    
    print_step_header("FEATURE ENGINEERING", 2)
    
    # Extraer configuración
    financial_params = config.get('financial_params', {})
    features_config = config.get('features', {})
    data_params = config.get('data_params', {})
    
    risk_free_rate = financial_params.get('risk_free_rate', RISK_FREE_RATE)
    momentum_days = config.get('momentum_score', {}).get('momentum_days', 126)
    benchmark = data_params.get('benchmark_ticker', 'SPY')
    
    print_info(f"Calculando 21 métricas financieras para {len(df_prices.columns)} activos...")
    print_info(f"Risk-free rate: {risk_free_rate:.1%}")
    print_info(f"Benchmark: {benchmark}")
    
    # Calcular features
    df_features = calculate_features_matrix(
        df_prices=df_prices,
        benchmark_ticker=benchmark,
        risk_free_rate=risk_free_rate,
        momentum_days=momentum_days
    )
    
    print_success(f"Features calculadas para {len(df_features)} activos")
    print_info(f"Métricas: {list(df_features.columns)}")
    
    return df_features


# Mantener compatibilidad con función anterior
def calculate_all_features(prices: pd.Series,
                           benchmark_returns: pd.Series,
                           risk_free_rate: float = RISK_FREE_RATE) -> dict:
    """
    Calcular métricas básicas para un activo (compatibilidad).
    
    Args:
        prices: Serie de precios del activo
        benchmark_returns: Serie de retornos del benchmark
        risk_free_rate: Tasa libre de riesgo
        
    Returns:
        Diccionario con métricas básicas
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
