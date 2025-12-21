# ================================================
# Backtesting Module
# ================================================
"""
Simulación y backtesting de estrategias de inversión.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime


def run_backtest(prices: pd.DataFrame,
                 target_weights: Dict[str, float],
                 initial_capital: float = 100000,
                 rebalance_freq: str = 'ME',
                 transaction_cost: float = 0.001) -> pd.Series:
    """
    Ejecutar backtesting con rebalanceo periódico.
    
    Args:
        prices: DataFrame de precios (índice = fecha, columnas = tickers)
        target_weights: Diccionario ticker -> peso objetivo
        initial_capital: Capital inicial
        rebalance_freq: Frecuencia de rebalanceo ('ME' = mensual, 'QE' = trimestral)
        transaction_cost: Costo de transacción (0.001 = 0.1%)
        
    Returns:
        Serie con valor del portfolio por fecha
    """
    # Validar que todos los tickers estén en prices
    tickers = list(target_weights.keys())
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Tickers no encontrados en prices: {missing}")
    
    prices = prices[tickers].copy()
    
    # Fechas de rebalanceo
    rebalance_dates = prices.resample(rebalance_freq).last().index
    
    portfolio_values = []
    dates = []
    
    # Estado inicial
    cash = initial_capital
    holdings = {ticker: 0.0 for ticker in tickers}
    
    for date in prices.index:
        # Calcular valor actual
        holdings_value = sum(holdings[t] * prices.loc[date, t] for t in tickers)
        current_value = cash + holdings_value
        
        # Rebalancear si es fecha de rebalanceo
        if date in rebalance_dates:
            # Vender todo (con costo de transacción)
            for ticker in tickers:
                if holdings[ticker] > 0:
                    sell_value = holdings[ticker] * prices.loc[date, ticker]
                    cash += sell_value * (1 - transaction_cost)
                    holdings[ticker] = 0.0
            
            # Recalcular valor después de vender
            current_value = cash
            
            # Comprar según pesos objetivo
            for ticker, weight in target_weights.items():
                investment = current_value * weight
                buy_cost = investment * (1 + transaction_cost)
                
                if buy_cost <= cash:
                    shares = investment / prices.loc[date, ticker]
                    holdings[ticker] = shares
                    cash -= buy_cost
        
        # Recalcular valor final del día
        holdings_value = sum(holdings[t] * prices.loc[date, t] for t in tickers)
        final_value = cash + holdings_value
        
        portfolio_values.append(final_value)
        dates.append(date)
    
    return pd.Series(portfolio_values, index=dates, name='portfolio_value')


def calculate_benchmark(prices: pd.Series,
                        initial_capital: float = 100000) -> pd.Series:
    """
    Calcular equity curve del benchmark (buy & hold).
    
    Args:
        prices: Serie de precios del benchmark
        initial_capital: Capital inicial
        
    Returns:
        Serie con valor del benchmark
    """
    return initial_capital * (prices / prices.iloc[0])


def calculate_metrics(equity_curve: pd.Series,
                      risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calcular métricas de rendimiento del portfolio.
    
    Args:
        equity_curve: Serie con valores del portfolio
        risk_free_rate: Tasa libre de riesgo anual
        
    Returns:
        Diccionario con métricas
    """
    returns = equity_curve.pct_change().dropna()
    
    # Retorno total
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    # Calcular años
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25
    
    # Retorno anualizado
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Volatilidad anualizada
    volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Maximum Drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    # Calmar Ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win Rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_days': len(equity_curve),
        'n_years': years
    }


def compare_portfolios(portfolio_equity: pd.Series,
                       benchmark_equity: pd.Series,
                       risk_free_rate: float = 0.02) -> pd.DataFrame:
    """
    Comparar métricas entre portfolio y benchmark.
    
    Args:
        portfolio_equity: Equity curve del portfolio
        benchmark_equity: Equity curve del benchmark
        risk_free_rate: Tasa libre de riesgo
        
    Returns:
        DataFrame con comparación
    """
    portfolio_metrics = calculate_metrics(portfolio_equity, risk_free_rate)
    benchmark_metrics = calculate_metrics(benchmark_equity, risk_free_rate)
    
    comparison = pd.DataFrame({
        'Portfolio': portfolio_metrics,
        'Benchmark': benchmark_metrics
    })
    
    # Calcular diferencia
    comparison['Difference'] = comparison['Portfolio'] - comparison['Benchmark']
    
    return comparison


def calculate_rolling_metrics(equity_curve: pd.Series,
                              window: int = 252) -> pd.DataFrame:
    """
    Calcular métricas rolling.
    
    Args:
        equity_curve: Serie con valores del portfolio
        window: Ventana en días (default: 1 año)
        
    Returns:
        DataFrame con métricas rolling
    """
    returns = equity_curve.pct_change().dropna()
    
    rolling_return = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / rolling_vol
    
    # Rolling drawdown
    rolling_max = equity_curve.rolling(window).max()
    rolling_drawdown = (equity_curve - rolling_max) / rolling_max
    
    return pd.DataFrame({
        'rolling_return': rolling_return,
        'rolling_volatility': rolling_vol,
        'rolling_sharpe': rolling_sharpe,
        'rolling_drawdown': rolling_drawdown
    })


def format_metrics_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Formatear métricas para presentación.
    
    Args:
        metrics: Diccionario de métricas
        
    Returns:
        DataFrame formateado
    """
    formatted = {}
    
    pct_metrics = ['total_return', 'annual_return', 'volatility', 'max_drawdown', 'win_rate']
    
    for key, value in metrics.items():
        if key in pct_metrics:
            formatted[key] = f"{value:.2%}"
        elif key in ['n_days', 'n_years']:
            formatted[key] = f"{value:.0f}" if key == 'n_days' else f"{value:.1f}"
        else:
            formatted[key] = f"{value:.2f}"
    
    return pd.DataFrame.from_dict(formatted, orient='index', columns=['Value'])
