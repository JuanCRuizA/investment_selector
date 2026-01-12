# ================================================
# Backtesting Module
# ================================================
"""
Simulación y backtesting de estrategias de inversión.
Incluye Buy & Hold y estrategias con rebalanceo.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
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


# ================================================
# FUNCIONES NUEVAS PARA PIPELINE PRODUCTIVO
# ================================================

def simular_buy_and_hold(df_prices: pd.DataFrame,
                         tickers: List[str],
                         capital_inicial: float = 10000,
                         costo_roundtrip: float = 0.001,
                         pesos: Dict[str, float] = None,
                         risk_free_rate: float = 0.045) -> Dict:
    """
    Simular estrategia Buy & Hold con costos de transacción.
    
    Esta función replica la lógica del notebook 04 para backtesting.
    
    Args:
        df_prices: DataFrame con precios históricos
        tickers: Lista de tickers a comprar
        capital_inicial: Capital inicial en USD
        costo_roundtrip: Costo total de transacción (compra + venta)
        pesos: Diccionario con pesos por ticker (None = equiponderado)
        risk_free_rate: Tasa libre de riesgo para métricas
        
    Returns:
        Diccionario con resultados del backtest
    """
    # Filtrar tickers disponibles
    tickers_disponibles = [t for t in tickers if t in df_prices.columns]
    
    if len(tickers_disponibles) == 0:
        raise ValueError("No hay tickers disponibles en los datos de precios")
    
    # Definir pesos (equiponderado si no se especifica)
    if pesos is None:
        peso_por_activo = 1 / len(tickers_disponibles)
        pesos = {t: peso_por_activo for t in tickers_disponibles}
    
    # ==========================================
    # FASE 1: COMPRA INICIAL
    # ==========================================
    
    precios_iniciales = df_prices[tickers_disponibles].iloc[0]
    
    # Capital disponible después de costos de entrada
    costo_entrada = costo_roundtrip / 2
    capital_despues_costos = capital_inicial * (1 - costo_entrada)
    
    # Calcular posiciones
    posiciones = {}
    for ticker in tickers_disponibles:
        capital_asignado = capital_despues_costos * pesos[ticker]
        precio_compra = precios_iniciales[ticker]
        n_acciones = capital_asignado / precio_compra
        posiciones[ticker] = n_acciones
    
    # ==========================================
    # FASE 2: EQUITY CURVE DIARIO
    # ==========================================
    
    equity_curve = pd.Series(index=df_prices.index, dtype=float)
    
    for fecha in df_prices.index:
        valor_dia = sum(
            posiciones[ticker] * df_prices.loc[fecha, ticker]
            for ticker in tickers_disponibles
        )
        equity_curve[fecha] = valor_dia
    
    # ==========================================
    # FASE 3: VENTA FINAL (costos de salida)
    # ==========================================
    
    costo_salida = costo_roundtrip / 2
    valor_final_bruto = equity_curve.iloc[-1]
    valor_final_neto = valor_final_bruto * (1 - costo_salida)
    equity_curve.iloc[-1] = valor_final_neto
    
    # ==========================================
    # FASE 4: MÉTRICAS
    # ==========================================
    
    retornos_diarios = equity_curve.pct_change().dropna()
    
    # Retornos
    retorno_total = (valor_final_neto - capital_inicial) / capital_inicial
    dias_trading = len(df_prices)
    retorno_anualizado = (1 + retorno_total) ** (252 / dias_trading) - 1
    
    # Volatilidad
    volatilidad_diaria = retornos_diarios.std()
    volatilidad_anualizada = volatilidad_diaria * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe_ratio = (retorno_anualizado - risk_free_rate) / volatilidad_anualizada if volatilidad_anualizada > 0 else 0
    
    # Sortino Ratio
    retornos_negativos = retornos_diarios[retornos_diarios < 0]
    downside_std = retornos_negativos.std() * np.sqrt(252) if len(retornos_negativos) > 0 else 0
    sortino_ratio = (retorno_anualizado - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    # Maximum Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar_ratio = retorno_anualizado / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'equity_curve': equity_curve,
        'drawdown': drawdown,
        'retornos_diarios': retornos_diarios,
        'capital_inicial': capital_inicial,
        'valor_final': valor_final_neto,
        'retorno_total': retorno_total,
        'retorno_anualizado': retorno_anualizado,
        'volatilidad_anualizada': volatilidad_anualizada,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'posiciones': posiciones,
        'tickers': tickers_disponibles,
        'dias_trading': dias_trading
    }


def run_backtests_all_profiles(portfolios: Dict[str, pd.DataFrame],
                                df_prices_test: pd.DataFrame,
                                benchmark_ticker: str = 'SPY',
                                capital_inicial: float = 10000,
                                costo_roundtrip: float = 0.001,
                                risk_free_rate: float = 0.045) -> Dict[str, Dict]:
    """
    Ejecutar backtests para todos los perfiles.
    
    Args:
        portfolios: Diccionario {perfil: df_portfolio}
        df_prices_test: DataFrame de precios del período de prueba
        benchmark_ticker: Ticker del benchmark
        capital_inicial: Capital inicial
        costo_roundtrip: Costo de transacción
        risk_free_rate: Tasa libre de riesgo
        
    Returns:
        Diccionario con resultados por perfil
    """
    results = {}
    
    # Backtest del benchmark (una sola vez)
    benchmark_result = simular_buy_and_hold(
        df_prices=df_prices_test,
        tickers=[benchmark_ticker],
        capital_inicial=capital_inicial,
        costo_roundtrip=costo_roundtrip,
        risk_free_rate=risk_free_rate
    )
    
    for perfil_name, df_portfolio in portfolios.items():
        print(f"   📊 Backtesting: {perfil_name.capitalize()}")
        
        # Obtener tickers del portafolio
        if 'ticker' in df_portfolio.columns:
            tickers = df_portfolio['ticker'].tolist()
        else:
            tickers = df_portfolio.index.tolist()
        
        # Ejecutar backtest
        portfolio_result = simular_buy_and_hold(
            df_prices=df_prices_test,
            tickers=tickers,
            capital_inicial=capital_inicial,
            costo_roundtrip=costo_roundtrip,
            risk_free_rate=risk_free_rate
        )
        
        results[perfil_name] = {
            'portfolio': portfolio_result,
            'benchmark': benchmark_result,
            'alpha': portfolio_result['retorno_total'] - benchmark_result['retorno_total']
        }
        
        print(f"      ✅ Retorno: {portfolio_result['retorno_total']:.2%} | Alpha: {results[perfil_name]['alpha']:.2%}")
    
    return results


def generate_backtest_metrics_df(backtest_result: Dict,
                                  perfil_name: str) -> pd.DataFrame:
    """
    Generar DataFrame de métricas de backtest en formato exportable.
    
    Args:
        backtest_result: Resultado de backtest (portfolio + benchmark)
        perfil_name: Nombre del perfil
        
    Returns:
        DataFrame con métricas
    """
    port = backtest_result['portfolio']
    bench = backtest_result['benchmark']
    
    metrics = {
        'Metrica': [
            'Retorno_Total',
            'Retorno_Anualizado',
            'Volatilidad_Anualizada',
            'Sharpe_Ratio',
            'Sortino_Ratio',
            'Max_Drawdown',
            'Calmar_Ratio'
        ],
        f'Portafolio_{perfil_name.capitalize()}': [
            port['retorno_total'],
            port['retorno_anualizado'],
            port['volatilidad_anualizada'],
            port['sharpe_ratio'],
            port['sortino_ratio'],
            port['max_drawdown'],
            port['calmar_ratio']
        ],
        'SPY_Benchmark': [
            bench['retorno_total'],
            bench['retorno_anualizado'],
            bench['volatilidad_anualizada'],
            bench['sharpe_ratio'],
            bench['sortino_ratio'],
            bench['max_drawdown'],
            bench['calmar_ratio']
        ]
    }
    
    return pd.DataFrame(metrics)


def run_backtest_pipeline(portfolios: Dict[str, pd.DataFrame],
                           df_prices_test: pd.DataFrame,
                           config: Dict) -> Dict[str, Dict]:
    """
    Ejecutar pipeline completo de backtesting.
    
    Args:
        portfolios: Diccionario con portafolios por perfil
        df_prices_test: DataFrame de precios de prueba
        config: Configuración del pipeline
        
    Returns:
        Diccionario con resultados de backtest
    """
    from .utils import print_step_header, print_success, print_info
    
    print_step_header("BACKTESTING", 4.1)
    
    # Extraer configuración
    backtest_config = config.get('backtesting', {})
    data_params = config.get('data_params', {})
    
    capital_inicial = backtest_config.get('initial_capital', 10000)
    costo_roundtrip = backtest_config.get('total_cost_roundtrip', 0.001)
    risk_free_rate = backtest_config.get('risk_free_rate_backtest', 0.045)
    benchmark = data_params.get('benchmark_ticker', 'SPY')
    
    print_info(f"Capital inicial: ${capital_inicial:,}")
    print_info(f"Costo round-trip: {costo_roundtrip:.2%}")
    print_info(f"Benchmark: {benchmark}")
    print_info(f"Período: {df_prices_test.index.min().strftime('%Y-%m-%d')} a {df_prices_test.index.max().strftime('%Y-%m-%d')}")
    
    # Ejecutar backtests
    results = run_backtests_all_profiles(
        portfolios=portfolios,
        df_prices_test=df_prices_test,
        benchmark_ticker=benchmark,
        capital_inicial=capital_inicial,
        costo_roundtrip=costo_roundtrip,
        risk_free_rate=risk_free_rate
    )
    
    print_success(f"Backtests completados: {len(results)} perfiles")
    
    return results
